import os

print("Step 1: Generating masks...")
os.system("python seg/generate_masks.py")

print("Step 2: Applying masks...")
os.system("python scripts/apply_mask.py")

# print("Step 3: Extract features...")
# os.system("python scripts/extract_features.py")

# print("Step 4: Compress features...")
# os.system("python scripts/compress_features.py")

print("✅ Pipeline ready for GAN training")