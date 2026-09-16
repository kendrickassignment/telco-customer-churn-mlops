"""TFX Transform module for Telco Customer Churn prediction.

This module defines the preprocessing function used by the TFX Transform
component. It applies feature engineering to raw input data including
scaling numerical features and encoding categorical features.
"""
import tensorflow as tf
import tensorflow_transform as tft

# Feature constants
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
LABEL_KEY = "Churn"
FEATURE_KEYS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Vocabulary sizes for categorical features
VOCAB_SIZE = 20
OOV_SIZE = 5


def transformed_name(key):
    """Generate the name for a transformed feature.

    Args:
        key: Original feature name.

    Returns:
        Transformed feature name with '_xf' suffix.
    """
    return f"{key}_xf"


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features.

    This function is used by the TFX Transform component to create
    a TensorFlow graph that applies feature engineering transformations.

    Args:
        inputs: Dictionary of input tensors keyed by feature name.

    Returns:
        Dictionary of transformed tensors keyed by transformed feature name.
    """
    outputs = {}

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            _fill_missing_numerical(inputs[feature])
        )

    outputs[transformed_name("SeniorCitizen")] = _fill_missing_numerical(
        inputs["SeniorCitizen"]
    )

    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            _fill_missing_categorical(inputs[feature]),
            top_k=VOCAB_SIZE,
            num_oov_buckets=OOV_SIZE,
            vocab_filename=transformed_name(feature),
        )

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(_fill_missing_categorical(inputs[LABEL_KEY]), "Yes"),
        tf.int64,
    )

    return outputs


def _fill_missing_numerical(tensor):
    """Replace missing numerical values with zero.

    Args:
        tensor: Input sparse or dense tensor.

    Returns:
        Dense tensor with missing values filled.
    """
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse.to_dense(tensor, default_value=0.0)
    return tf.squeeze(tensor, axis=1)


def _fill_missing_categorical(tensor):
    """Replace missing categorical values with empty string.

    Args:
        tensor: Input sparse or dense tensor.

    Returns:
        Dense tensor with missing values filled.
    """
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse.to_dense(tensor, default_value="")
    return tf.squeeze(tensor, axis=1)