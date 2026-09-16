"""TFX Tuner module for hyperparameter tuning of Churn prediction model.

This module defines the tuner function used by the TFX Tuner component
to perform automated hyperparameter search using Keras Tuner.
"""
import os

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.public import tfxio as tfxio_module

from modules.transform_module import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    OOV_SIZE,
    transformed_name,
)

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64


def _build_keras_model_tuner(hp, tf_transform_output):
    """Build a tunable Keras model for hyperparameter search."""
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
        embedding_dim = hp.Choice(
            f"embed_dim_{feature}", values=[4, 8, 16], default=8
        )
        embedding = layers.Embedding(
            input_dim=vocab_size + OOV_SIZE + 1,
            output_dim=embedding_dim,
        )(feat_input)
        embedding = layers.Flatten()(embedding)
        encoded_features.append(embedding)

    concatenated = layers.Concatenate()(encoded_features)
    x = concatenated

    num_layers = hp.Int("num_layers", min_value=2, max_value=4, default=3)
    for i in range(num_layers):
        units = hp.Choice(f"units_{i}", values=[32, 64, 128, 256], default=64)
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        dropout_rate = hp.Float(
            f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1, default=0.3
        )
        x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_features, outputs=output)

    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-3
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size):
    """Create input function for tuning data."""
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio_module.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=transformed_name(LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    )
    return dataset


def tuner_fn(fn_args: FnArgs):
    """Define the tuner configuration for hyperparameter search.

    Returns:
        TunerFnResult namedtuple with tuner object and fit kwargs.
    """
    # Resolve transform_output path
    transform_output_path = fn_args.transform_output
    if transform_output_path is None:
        transform_output_path = fn_args.transform_graph_path

    tf_transform_output = tft.TFTransformOutput(transform_output_path)

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

    # Handle None working_dir
    working_dir = fn_args.working_dir
    if working_dir is None:
        working_dir = os.path.join(os.getcwd(), "tuner_working_dir")
    os.makedirs(working_dir, exist_ok=True)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: _build_keras_model_tuner(
            hp, tf_transform_output
        ),
        objective=kt.Objective("val_binary_accuracy", direction="max"),
        max_trials=6,
        executions_per_trial=1,
        directory=working_dir,
        project_name="churn_tuner",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 5,
        },
    )