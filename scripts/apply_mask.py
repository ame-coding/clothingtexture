import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_DIR = os.path.join(BASE_DIR, "data", "fashion_raw")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
OUT_DIR = os.path.join(BASE_DIR, "data", "masked")

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# RECURSIVE PROCESS
# =========================
for root, _, files in os.walk(IMG_DIR):
    for file in files:
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(root, file)

        rel_path = os.path.relpath(img_path, IMG_DIR)
        mask_path = os.path.join(MASK_DIR, rel_path)
        out_path = os.path.join(OUT_DIR, rel_path)

        try:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = mask / 255.0

            masked = img * mask[:, :, None]

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, masked.astype("uint8"))

        except Exception as e:
            print(f"Skipped: {img_path} → {e}")

print("🎉 Masked images created")