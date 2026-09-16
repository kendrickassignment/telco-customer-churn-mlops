"""TFX Trainer module for Telco Customer Churn prediction.

This module defines the model architecture and training logic used by the
TFX Trainer component. It builds a DNN classifier with embedding layers
for categorical features and dense layers for numerical features.
"""
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx_bsl.public import tfxio as tfxio_module
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.transform_module import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    OOV_SIZE,
    transformed_name,
)

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EPOCHS = 10


def _build_keras_model(tf_transform_output):
    """Build a Keras DNN model for churn prediction.

    Architecture:
        - Numerical inputs â†’ directly concatenated
        - SeniorCitizen â†’ cast to float, concatenated
        - Categorical inputs â†’ Embedding â†’ Flatten â†’ concatenated
        - Dense(128) â†’ BN â†’ Dropout â†’ Dense(64) â†’ BN â†’ Dropout â†’ Dense(32)
        - Output: Dense(1, sigmoid)

    Args:
        tf_transform_output: TFTransformOutput for feature metadata.

    Returns:
        Compiled Keras model.
    """
    input_features = []
    encoded_features = []

    for feature in NUMERICAL_FEATURES:
        feat_input = layers.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )
        input_features.append(feat_input)
        encoded_features.append(feat_input)

    senior_input = layers.Input(
        shape=(1,), name=transformed_name("SeniorCitizen"), dtype=tf.int64
    )
    input_features.append(senior_input)
    senior_float = tf.cast(senior_input, tf.float32)
    encoded_features.append(senior_float)

    for feature in CATEGORICAL_FEATURES:
        feat_input = layers.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.int64
        )
        input_features.append(feat_input)

        vocab_size = tf_transform_output.vocabulary_size_by_name(
            transformed_name(feature)
        )
        embedding = layers.Embedding(
            input_dim=vocab_size + OOV_SIZE + 1,
            output_dim=8,
        )(feat_input)
        embedding = layers.Flatten()(embedding)
        encoded_features.append(embedding)

    concatenated = layers.Concatenate()(encoded_features)

    x = layers.Dense(128, activation="relu")(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_features, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Create a serving function that applies transform then predicts.

    This function creates a signature that:
    1. Receives raw serialized tf.Examples
    2. Applies the Transform preprocessing graph
    3. Runs the model prediction

    Args:
        model: Trained Keras model.
        tf_transform_output: TFTransformOutput for the transform graph.

    Returns:
        A function that can be used as a serving signature.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Parse raw examples, transform, and predict."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop("Churn")

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {"churn_probability": outputs}

    return serve_tf_examples_fn


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size):
    """Create input function for training/eval data.

    Args:
        file_pattern: File pattern for input data.
        data_accessor: DataAccessor for reading data.
        tf_transform_output: TFTransformOutput for feature metadata.
        batch_size: Batch size for the dataset.

    Returns:
        tf.data.Dataset of (features, labels) tuples.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio_module.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=transformed_name(LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    )
    return dataset


def run_fn(fn_args: FnArgs):
    """Train the model using TFX Trainer component.

    Args:
        fn_args: FnArgs object containing training arguments from TFX.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        TRAIN_BATCH_SIZE,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        EVAL_BATCH_SIZE,
    )

    model = _build_keras_model(tf_transform_output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures,
    )