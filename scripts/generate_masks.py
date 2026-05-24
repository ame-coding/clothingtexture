import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from u2net import U2NET

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "data", "fashion_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "masks")
MODEL_PATH = os.path.join(BASE_DIR, "models", "u2net.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = U2NET(3, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ U2Net loaded")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

def normalize(mask):
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

# =========================
# RECURSIVE IMAGE LOADING
# =========================
image_list = []

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_list.append(os.path.join(root, file))

print(f"Found {len(image_list)} images")

# =========================
# PROCESS
# =========================
for i, img_path in enumerate(image_list):

    try:
        image = Image.open(img_path).convert("RGB")
        original_size = image.size

        inp = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            d1, *_ = model(inp)

        mask = d1[:, 0, :, :].squeeze().cpu().numpy()
        mask = normalize(mask)
        mask = (mask * 255).astype(np.uint8)

        mask = cv2.resize(mask, original_size)

        # preserve folder structure
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        save_path = os.path.join(OUTPUT_DIR, rel_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)

        if i % 50 == 0:
            print(f"[{i}/{len(image_list)}] processed")

    except Exception as e:
        print(f"Skipped: {img_path} → {e}")

print("🎉 Masks generated")