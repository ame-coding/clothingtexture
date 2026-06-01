import os
import cv2
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--uuid", type=str, required=True)

# Parse command line
args = parser.parse_args()

uid = args.uuid


ROOT = Path(__file__).resolve().parent.parent

mask_path = ROOT / f"outputs/{uid}_mask.jpg"
out_path = ROOT / f"server/finished/{uid}_final.jpg"
img_path=ROOT / f"server/uploads/{uid}_texture.jpg"


try:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = mask / 255.0

            masked = img * mask[:, :, None]


            cv2.imwrite(out_path, masked.astype("uint8"))

except Exception as e:
            print(f"Skipped: {img_path} → {e}")

print("🎉 Masked images created")