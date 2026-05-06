import os
import numpy as np
import cv2
import torch
import gc
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

# ========================
# CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "textures")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

# ========================
# LOAD MODEL
# ========================
weights = VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights).features[:24]
model = model.to(device)
model.eval()

print("✅ VGG16 loaded")

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========================
# LOAD IMAGES (RECURSIVE)
# ========================
image_list = []

for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_list.append(os.path.join(root, file))

print(f"📂 Found {len(image_list)} images")

if len(image_list) == 0:
    raise ValueError("❌ No images found.")

# ========================
# CREATE MEMMAP
# ========================
num_images = len(image_list)
feature_dim = 100352  # VGG16 output

features_path = os.path.join(FEATURE_DIR, "features.dat")

features_memmap = np.memmap(
    features_path,
    dtype="float32",
    mode="w+",
    shape=(num_images, feature_dim)
)

paths = []

# ========================
# EXTRACT FEATURES
# ========================
for i, img_path in enumerate(tqdm(image_list, desc="Extracting")):

    try:
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipped unreadable: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        fmap = model(img)

        feat = fmap.view(-1).cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        features_memmap[i] = feat.astype(np.float32)
        paths.append(img_path)

        # 🔥 MEMORY CLEANUP
        del img, fmap, feat

        if i % 50 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Skipped: {img_path} → {e}")

# ========================
# SAVE PATHS
# ========================
features_memmap.flush()

np.save(os.path.join(FEATURE_DIR, "paths.npy"), np.array(paths))

print("\n🎉 Feature extraction complete (SAFE)")
print(f"Saved to: {features_path}")
print(f"Shape: ({num_images}, {feature_dim})")