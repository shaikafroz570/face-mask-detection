# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------
# Hyperparameters
# ----------------------------
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# ----------------------------
# Dataset Path
# ----------------------------
DIRECTORY = "dataset"

CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Loading images...")

data = []
labels = []

# ----------------------------
# Load Dataset
# ----------------------------
for category in CATEGORIES:

    path = os.path.join(DIRECTORY, category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# ----------------------------
# Convert Labels
# ----------------------------
lb = LabelBinarizer()

labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# ----------------------------
# Split Dataset
# ----------------------------
(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# ----------------------------
# Data Augmentation
# ----------------------------
aug = ImageDataGenerator(

    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ----------------------------
# Load MobileNetV2
# ----------------------------
print("[INFO] Loading MobileNetV2...")

baseModel = MobileNetV2(

    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224,224,3))
)

# ----------------------------
# Build Head Model
# ----------------------------
headModel = baseModel.output

headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)

headModel = Dense(128, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(2, activation="softmax")(headModel)

# ----------------------------
# Final Model
# ----------------------------
model = Model(
    inputs=baseModel.input,
    outputs=headModel
)

# ----------------------------
# Freeze Base Layers
# ----------------------------
for layer in baseModel.layers:
    layer.trainable = False

# ----------------------------
# Compile
# ----------------------------
print("[INFO] Compiling model...")

opt = Adam(learning_rate=INIT_LR)

model.compile(

    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# ----------------------------
# Train
# ----------------------------
print("[INFO] Training network...")

H = model.fit(

    aug.flow(trainX, trainY, batch_size=BS),

    validation_data=(testX, testY),

    epochs=EPOCHS
)

# ----------------------------
# Predict
# ----------------------------
print("[INFO] Evaluating network...")

predIdxs = model.predict(testX, batch_size=BS)

predIdxs = np.argmax(predIdxs, axis=1)

print(

    classification_report(

        testY.argmax(axis=1),
        predIdxs,
        target_names=lb.classes_
    )
)

# ----------------------------
# Save Model
# ----------------------------
print("[INFO] Saving model...")

model.save("mask_detector.keras")

print("[INFO] Model Saved Successfully!")

# ----------------------------
# Plot Accuracy & Loss
# ----------------------------
plt.style.use("ggplot")

plt.figure()

N = EPOCHS

plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0,N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss & Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Loss / Accuracy")

plt.legend(loc="lower left")

plt.savefig("plot.png")

plt.show()