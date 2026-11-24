"""
Revivo - train a waste classifier (9 Kaggle classes + your own extra classes, e.g. Religious Waste)

Kaggle dataset: https://www.kaggle.com/datasets/adithyachalla/waste-classification

HOW TO ADD RELIGIOUS WASTE (the Kaggle dataset has no such class, so YOU must supply images):
  Create this folder structure next to this script:

      extra_classes/
          Religious Waste/
              img001.jpg
              img002.jpg
              ... (aim for 150-300+ photos)

  Good photos to collect: clay / mitti idols (Ganesh, Durga, Lakshmi etc.), Plaster of Paris idols,
  metal/stone idols, garlands and puja flowers, diyas, used puja items. Use your own photos or images
  you are allowed to use. Vary angle, light, background, and include "waste-like" settings
  (idols near a bucket/immersion tank, flowers in a pile). Too few images = unreliable detection.

Run in Google Colab (GPU: Runtime > Change runtime type > GPU):
    !pip install -q kagglehub tensorflowjs tf_keras
    # upload extra_classes/ and this script to Colab, then:
    !python train_model.py
Then download the generated  model/  folder and put it next to index.html.
"""
import os, json, shutil
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # needed so TensorFlow.js can load the model
import tensorflow as tf

DATA_DIR = None            # e.g. "/content/waste" to skip the kagglehub download
EXTRA_DIR = "extra_classes"   # your own classes (one sub-folder per class)
MERGED = "merged_dataset"
IMG = 224
BATCH = 32
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

if DATA_DIR is None:
    import kagglehub
    DATA_DIR = kagglehub.dataset_download("adithyachalla/waste-classification")

# find the folder that directly contains the Kaggle class folders (1-Cardboard ... 9-Vegetation)
root = None
for r, dirs, _ in os.walk(DATA_DIR):
    if any(d.startswith("9-") or d.lower() == "vegetation" for d in dirs):
        root = r
        break
assert root, "Could not find class folders in " + DATA_DIR
print("Kaggle dataset root:", root)


def count_images(path):
    return sum(1 for f in os.listdir(path) if f.lower().endswith(EXTS))


def link(src, dst):
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


# merge Kaggle classes + extra classes into one folder
shutil.rmtree(MERGED, ignore_errors=True)
os.makedirs(MERGED)
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if os.path.isdir(p):
        link(p, os.path.join(MERGED, d))

if os.path.isdir(EXTRA_DIR):
    for d in sorted(os.listdir(EXTRA_DIR)):
        p = os.path.join(EXTRA_DIR, d)
        if os.path.isdir(p):
            n = count_images(p)
            print(f"Extra class '{d}': {n} images")
            if n < 50:
                print(f"  WARNING: only {n} images - detection of '{d}' will be unreliable. Add more.")
            link(p, os.path.join(MERGED, d))
else:
    print("WARNING: no extra_classes/ folder found - training only the 9 Kaggle classes (no Religious Waste).")

train = tf.keras.utils.image_dataset_from_directory(
    MERGED, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH, follow_links=True)
val = tf.keras.utils.image_dataset_from_directory(
    MERGED, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG, IMG), batch_size=BATCH, follow_links=True)

raw_names = train.class_names                       # alphabetical order
labels = []
for n in raw_names:
    l = n.split("-", 1)[1] if n[:1].isdigit() and "-" in n else n
    labels.append(l.replace("_", " ").strip())
print("Classes:", labels)

# class weights so smaller classes (e.g. Religious Waste) are not ignored
counts = [count_images(os.path.join(MERGED, n)) for n in raw_names]
total = sum(counts)
class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
print("Images per class:", dict(zip(labels, counts)))

AUTOTUNE = tf.data.AUTOTUNE
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
])
prep = tf.keras.applications.mobilenet_v2.preprocess_input   # scales to [-1, 1]
# augmentation + preprocessing live in the data pipeline (not the saved model),
# so the exported model is TensorFlow.js compatible. The website applies x/127.5-1.
train = train.shuffle(1000).map(lambda x, y: (prep(aug(x, training=True)), y),
                                num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val = val.map(lambda x, y: (prep(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = tf.keras.Input((IMG, IMG, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
model = tf.keras.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8, class_weight=class_weight)

# fine-tune top layers
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=6, class_weight=class_weight)

loss, acc = model.evaluate(val)
print(f"Validation accuracy: {acc:.3f}")

os.makedirs("model", exist_ok=True)
with open("model/labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "model")
print("Done. Copy the 'model' folder next to index.html")
