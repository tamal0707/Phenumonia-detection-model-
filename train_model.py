import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Expected directory structure:
# dataset/chest_xray/train/NORMAL
# dataset/chest_xray/train/PNEUMONIA
# dataset/chest_xray/validation/NORMAL
# dataset/chest_xray/validation/PNEUMONIA

train_dir = "dataset/chest_xray/train"
validation_dir = "dataset/chest_xray/validation"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

inputs = layers.Input(shape=(128, 128, 1))
x = data_augmentation(inputs)
x = layers.Conv2D(3, (1, 1), padding="same")(x)

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)
base_model.trainable = False

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

model.save("pneumonia_model.keras")
print("Model saved as pneumonia_model.keras")
