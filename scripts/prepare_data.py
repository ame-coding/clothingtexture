import pandas as pd
import os
import shutil

CSV_PATH = r"D:\MajorProject26\data\masked\images.csv"
IMG_DIR = r"D:\MajorProject26\data\masked"
OUT_DIR = r"D:\MajorProject26\data\fashion_dataset\train"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    img_id = str(row["image"])
    label = str(row["label"])

    # ❌ skip garbage labels
    if label == "Not sure" or label == "nan":
        continue

    # 🔥 normalize label (VERY IMPORTANT)
    label = label.strip().lower()

    src = os.path.join(IMG_DIR, f"{img_id}.jpg")

    if not os.path.exists(src):
        continue

    dst_folder = os.path.join(OUT_DIR, label)
    os.makedirs(dst_folder, exist_ok=True)

    dst = os.path.join(dst_folder, f"{img_id}.jpg")
    shutil.copy(src, dst)

print("✅ All categories dataset created!")