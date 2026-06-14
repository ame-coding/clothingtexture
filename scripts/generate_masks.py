import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from u2net import U2NET
import argparse


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--uuid", type=str, required=True)

args = parser.parse_args()

uid = args.uuid


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH =ROOT / "models" 

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


img_path= INPUT_DIR/ f"clothinggen_{uid}.jpg"

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

    outpath = Path(OUTPUT_DIR) / f"{uid}_mask.jpg"
    cv2.imwrite(outpath, mask)


except Exception as e:
    print(f"Skipped: {img_path} → {e}")

print("🎉 Masks generated")