import numpy as np
import tensorflow as tf

# Disable TF GPU to avoid driver conflicts during numpy training
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import tensorflow_datasets as tfds

# Load MNIST from TensorFlow datasets
(ds_train, ds_test), ds_info = tfds.load(
    "mnist", split=["train", "test"], as_supervised=True, with_info=True
)


def normalize_img(image, label):
    """Normalize images to [0, 1] range"""
    return tf.cast(image, tf.float32) / 255.0, label


# Apply normalization
ds_train_normalized = ds_train.map(normalize_img)
ds_test_normalized = ds_test.map(normalize_img)

# Convert to numpy arrays - iterate once and collect both images and labels
train_images = []
train_labels = []
for image, label in ds_train_normalized:
    train_images.append(image.numpy())
    train_labels.append(label.numpy())

test_images = []
test_labels = []
for image, label in ds_test_normalized:
    test_images.append(image.numpy())
    test_labels.append(label.numpy())

# Stack into arrays
X_train_raw = np.stack(train_images)  # (60000, 28, 28, 1)
TRAINING_LABELS = np.stack(train_labels).reshape(-1, 1)  # (60000, 1)

X_test_raw = np.stack(test_images)  # (10000, 28, 28, 1)
TEST_LABELS = np.stack(test_labels).reshape(-1, 1)  # (10000, 1)

# Flatten images and add bias term
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)  # (60000, 784)
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)  # (10000, 784)


def prepend_bias(X):
    """Insert a column of 1s at position 0 for bias term"""
    bias = np.ones((X.shape[0], 1), dtype=np.float32)
    return np.concatenate([bias, X], axis=1)


# Add bias term
X_train = prepend_bias(X_train)  # (60000, 785)
X_test = prepend_bias(X_test)  # (10000, 785)


def encode_digit(Y, digit):
    """Vectorized binary encoding for a specific digit"""
    return (Y == digit).astype(np.float32)


# Create binary classifiers for each digit (0-9)
Y_train = []
Y_test = []

for digit in range(10):
    Y_train.append(encode_digit(TRAINING_LABELS, digit))
    Y_test.append(encode_digit(TEST_LABELS, digit))

# Ensure float32 for all arrays
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print(f"\nLoaded MNIST dataset from TensorFlow:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  Y_train: 10 binary classifiers, each shape {Y_train[0].shape}")
print(f"  Y_test: 10 binary classifiers, each shape {Y_test[0].shape}")
print(f"  Dataset info: {ds_info.splits}")
