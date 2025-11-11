import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

(ds_train, ds_test), ds_info = tfds.load(
    "mnist", split=["train", "test"], as_supervised=True, with_info=True
)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


# Apply normalization
ds_train_normalized = ds_train.map(normalize_img)
ds_test_normalized = ds_test.map(normalize_img)

# Convert to numpy arrays - extract both images and labels
X_train = np.array([image.numpy() for image, _ in ds_train_normalized])
Y_train = np.array([label.numpy() for _, label in ds_train_normalized])

X_test = np.array([image.numpy() for image, _ in ds_test_normalized])
Y_test = np.array([label.numpy() for _, label in ds_test_normalized])

# Flatten images and add bias term
X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
X_test = X_test.reshape(X_test.shape[0], -1)  # (10000, 784)

bias_train = np.ones((X_train.shape[0], 1), dtype=np.float32)
bias_test = np.ones((X_test.shape[0], 1), dtype=np.float32)

X_train = np.concatenate([X_train, bias_train], axis=1)  # (60000, 785)
X_test = np.concatenate([X_test, bias_test], axis=1)  # (10000, 785)

# Flatten labels to match main.py format
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# Create variables compatible with main.py
X = X_train  # Image data with bias
Y = Y_train  # Labels (flattened)

print("Dataset info:", ds_info)
print("Training images shape (with bias):", X_train.shape)
print("Training labels shape:", Y_train.shape)
print("Testing images shape (with bias):", X_test.shape)
print("Testing labels shape:", Y_test.shape)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Sample label:", Y[0])
