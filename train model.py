"""
Revivo - train a waste classifier on the Kaggle dataset
https://www.kaggle.com/datasets/adithyachalla/waste-classification

Run in Google Colab (free GPU: Runtime > Change runtime type > GPU):
    !pip install -q kagglehub tensorflowjs
    !python train_model.py
Then download the generated  model/  folder and put it next to index.html.
(If kagglehub asks for credentials, add your Kaggle API key, or upload the
dataset manually and set DATA_DIR below.)
"""
import os, json
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tfjs export needs Keras 2 (pip install tf_keras)
import tensorflow as tf

DATA_DIR = None          # e.g. "/content/waste" to skip kagglehub download
IMG = 224
BATCH = 32

if DATA_DIR is None:
    import kagglehub
    DATA_DIR = kagglehub.dataset_download("adithyachalla/waste-classification")

# find the folder that directly contains the class folders (1-Cardboard ... 9-Vegetation)
root = None
for r, dirs, _ in os.walk(DATA_DIR):
    if any(d.startswith("9-") or d.lower() == "vegetation" for d in dirs):
        root = r
        break
assert root, "Could not find class folders in " + DATA_DIR
print("Dataset root:", root)

train = tf.keras.utils.image_dataset_from_directory(
    root, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH)
val = tf.keras.utils.image_dataset_from_directory(
    root, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH)

raw_names = train.class_names                       # alphabetical order
labels = [n.split("-", 1)[1] if n[0].isdigit() else n for n in raw_names]
labels = [l.replace("_", " ").strip() for l in labels]
print("Classes:", labels)

AUTOTUNE = tf.data.AUTOTUNE
# augmentation lives in the data pipeline only (tfjs cannot load Random* layers)
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
])
train = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
train = train.shuffle(1000).prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

# model takes raw 0-255 pixels (the website feeds raw pixels)
inp = tf.keras.Input((IMG, IMG, 3))
x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
model = tf.keras.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)

# fine-tune top layers
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=6)

loss, acc = model.evaluate(val)
print(f"Validation accuracy: {acc:.3f}")

os.makedirs("model", exist_ok=True)
with open("model/labels.json", "w") as f:
    json.dump(labels, f)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "model")
print("Done. Copy the 'model' folder next to index.html")