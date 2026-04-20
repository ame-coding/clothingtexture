# scripts/verify_setup.py
# PURPOSE: Confirms everything is installed correctly and finds your DTD images

import os
import sys

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

# 1. Check Python version
print(f"Python version: {sys.version}")

# 2. Check all libraries
libraries = ["tensorflow", "numpy", "matplotlib", "PIL", "sklearn", "tqdm"]
for lib in libraries:
    try:
        __import__(lib if lib != "PIL" else "PIL.Image")
        print(f"  ✓ {lib}")
    except ImportError:
        print(f"  ✗ {lib} — NOT FOUND, run pip install again")

# 3. Check TensorFlow can see GPU (optional but useful)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU available: {'YES — ' + str(len(gpus)) + ' GPU(s)' if gpus else 'NO — will use CPU (slower but fine)'}")

# 4. Check folder structure
print("\n" + "=" * 50)
print("FOLDER STRUCTURE CHECK")
print("=" * 50)

base = r"D:\MajorProject26"
expected_folders = ["data", "datasets", "features", "models", "notebooks", "outputs", "scripts"]

for folder in expected_folders:
    path = os.path.join(base, folder)
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {status}  {folder}/")

# 5. Count DTD images
dtd_path = os.path.join(base, "datasets", "dtd", "images")
if os.path.exists(dtd_path):
    categories = os.listdir(dtd_path)
    total_images = 0
    for cat in categories:
        cat_path = os.path.join(dtd_path, cat)
        if os.path.isdir(cat_path):
            count = len([f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_images += count
    print(f"\n  ✓ DTD found: {len(categories)} categories, {total_images} total images")
else:
    print(f"\n  ✗ DTD not found at: {dtd_path}")
    print("    Make sure your DTD is at: data/dtd/images/<category>/")

print("\n✅ Setup check complete!")