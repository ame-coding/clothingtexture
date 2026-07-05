import os
import cv2

# =========================
# PATHS
# =========================
INPUT_DIR = r"C:\Users\nilot\MajorProject26\data\fashion_dataset\train"
OUTPUT_DIR = r"C:\Users\nilot\MajorProject26\data\fashion_dataset_128\train"
SIZE = 128
# =========================
# PROCESS
# =========================
for root, _, files in os.walk(INPUT_DIR):

    for file in files:

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        input_path = os.path.join(root, file)

        # preserve folder structure
        relative_path = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, relative_path)

        try:
            # read image
            img = cv2.imread(input_path)

            if img is None:
                print(f"Skipped unreadable: {input_path}")
                continue

            # resize
            img = cv2.resize(img, (SIZE, SIZE))

            # create output folder
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # save
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"Error: {input_path} → {e}")

print("✅ Dataset resized successfully")