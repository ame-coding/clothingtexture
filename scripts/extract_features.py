# scripts/extract_features.py
# PURPOSE: Loads VGG16, reads every DTD texture image,
#          extracts a feature vector per image, and saves them to /features

import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # progress bar

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ─── CONFIG ──────────────────────────────────────────────
BASE_DIR    = r"D:\MajorProject26"
DTD_DIR     = os.path.join(BASE_DIR, "datasets", "dtd", "images")
FEATURE_DIR = os.path.join(BASE_DIR, "features")
IMG_SIZE    = (224, 224)   # VGG16 expects 224x224
# ─────────────────────────────────────────────────────────


# ── STEP A: Build the feature extractor ──────────────────
print("Loading VGG16 model...")

base_model = VGG16(
    weights='imagenet',     # use pretrained ImageNet weights
    include_top=False,      # remove the final classification layers
    input_shape=(224, 224, 3)
)

# We extract features from 'block4_pool'
# This layer captures mid-level texture info (shapes, patterns, colors)
# Earlier layers = low-level (edges), Later layers = high-level (objects)
# block4_pool is the sweet spot for TEXTURE
feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('block4_pool').output
)

feature_extractor.trainable = False  # freeze — we're not training VGG16
print("✓ VGG16 loaded. Feature output shape per image: (14, 14, 512)\n")


# ── STEP B: Image preprocessing function ─────────────────
def preprocess_image(image_path):
    """
    Opens an image, resizes it to 224x224,
    and applies VGG16's expected normalization.
    """
    img = Image.open(image_path).convert('RGB')   # ensure 3 color channels
    img = img.resize(IMG_SIZE)                    # resize to 224x224
    arr = np.array(img, dtype=np.float32)         # convert to numpy array
    arr = np.expand_dims(arr, axis=0)             # shape: (1, 224, 224, 3)
    arr = preprocess_input(arr)                   # VGG16 normalization (subtracts ImageNet mean)
    return arr


# ── STEP C: Extract features from one image ──────────────
def extract_features(image_path):
    """
    Returns a flat 1D feature vector for a single image.
    VGG16 block4_pool output is (1, 14, 14, 512)
    After flattening → (100352,)
    """
    preprocessed = preprocess_image(image_path)
    feature_map = feature_extractor.predict(preprocessed, verbose=0)  # (1,14,14,512)
    return feature_map.flatten()  # → (100352,)


# ── STEP D: Loop through all DTD categories ──────────────
print("Starting feature extraction from DTD dataset...\n")

categories = sorted(os.listdir(DTD_DIR))

# Instead of storing everything in RAM, we save each category separately
os.makedirs(os.path.join(FEATURE_DIR, "by_category"), exist_ok=True)

all_labels = []
all_paths  = []

for category in categories:
    cat_path = os.path.join(DTD_DIR, category)
    if not os.path.isdir(cat_path):
        continue

    image_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing category: {category} ({len(image_files)} images)")

    cat_features = []

    for img_file in tqdm(image_files, desc=f"  {category}", ncols=70):
        img_path = os.path.join(cat_path, img_file)
        try:
            feat = extract_features(img_path).astype(np.float16)  # float16 = half the memory
            cat_features.append(feat)
            all_labels.append(category)
            all_paths.append(img_path)
        except Exception as e:
            print(f"  Skipped {img_file}: {e}")

    # Save this category's features immediately and free RAM
    cat_array = np.array(cat_features, dtype=np.float16)
    save_path = os.path.join(FEATURE_DIR, "by_category", f"{category}.npy")
    np.save(save_path, cat_array)
    print(f"  ✓ Saved {category}.npy — shape {cat_array.shape}")

    del cat_features, cat_array  # free RAM before next category


# ── STEP E: Merge all category files into one ────────────
print("\nMerging all categories...")

merged = []
for category in categories:
    path = os.path.join(FEATURE_DIR, "by_category", f"{category}.npy")
    if os.path.exists(path):
        merged.append(np.load(path))

features_array = np.concatenate(merged, axis=0)   # shape: (5640, 100352)
labels_array   = np.array(all_labels)
paths_array    = np.array(all_paths)

np.save(os.path.join(FEATURE_DIR, "dtd_features.npy"), features_array)
np.save(os.path.join(FEATURE_DIR, "dtd_labels.npy"),   labels_array)
np.save(os.path.join(FEATURE_DIR, "dtd_paths.npy"),    paths_array)

print(f"✓ dtd_features.npy → shape {features_array.shape}")
print(f"✓ dtd_labels.npy   → shape {labels_array.shape}")
print(f"✓ dtd_paths.npy    → shape {paths_array.shape}")
print(f"\n🎉 Feature extraction complete!")