"""
train.py — Handwritten Digit Classifier
Trains a CNN on the MNIST dataset and saves the model.

Usage:
    python train.py

Output:
    digit_model.keras  — saved Keras model
    training_history.png — accuracy/loss plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── Reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─── Load & Preprocess MNIST ──────────────────────────────────────────────────
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize to [0, 1] and add channel dim
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test,  -1)   # (10000, 28, 28, 1)

# One-hot encode labels
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes)

print(f"Train samples : {x_train.shape[0]}")
print(f"Test  samples : {x_test.shape[0]}")

# ─── Data Augmentation ────────────────────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),          # ±10% rotation (good for handwriting)
    layers.RandomZoom(0.1),              # ±10% zoom
    layers.RandomTranslation(0.1, 0.1), # slight shifts
], name="augmentation")

# ─── Model Architecture ───────────────────────────────────────────────────────
def build_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    # Augmentation (only active during training)
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="digit_cnn")


model = build_model()
model.summary()

# ─── Compile ──────────────────────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "digit_model_best.keras", monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
]

# ─── Train ────────────────────────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ─── Save Final Model ─────────────────────────────────────────────────────────
model.save("digit_model.keras")
print("Model saved → digit_model.keras")

# ─── Plot Training History ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="Train Acc")
axes[0].plot(history.history["val_accuracy"], label="Val Acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"CNN Training — Final Test Accuracy: {test_acc*100:.2f}%", fontsize=14)
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Training plot saved → training_history.png")
plt.show()
