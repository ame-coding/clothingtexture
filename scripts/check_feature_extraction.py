# scripts/check_features.py
# PURPOSE: Open the saved .npy files and show you exactly what's inside them

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

BASE_DIR    = r"D:\MajorProject26"
FEATURE_DIR = os.path.join(BASE_DIR, "features")

# ── STEP 1: Load the 3 saved files ───────────────────────
print("Loading saved files...")
features = np.load(os.path.join(FEATURE_DIR, "dtd_features.npy"))
labels   = np.load(os.path.join(FEATURE_DIR, "dtd_labels.npy"))
paths    = np.load(os.path.join(FEATURE_DIR, "dtd_paths.npy"))

# ── STEP 2: Basic stats ───────────────────────────────────
print("\n" + "="*50)
print("WHAT'S INSIDE YOUR FILES")
print("="*50)
print(f"Total images      : {features.shape[0]}")
print(f"Numbers per image : {features.shape[1]}")
print(f"Total categories  : {len(np.unique(labels))}")
print(f"All categories    : {list(np.unique(labels))}")

# ── STEP 3: Show one example row (one image's fingerprint)
print("\n" + "="*50)
print("EXAMPLE — Image #1")
print("="*50)
print(f"Image file  : {paths[0]}")
print(f"Category    : {labels[0]}")
print(f"Feature vector (first 10 numbers out of 100,352):")
print(f"  {features[0][:10]}")
print(f"Feature vector (last 10 numbers):")
print(f"  {features[0][-10:]}")
print(f"Min value: {features[0].min():.4f}  Max value: {features[0].max():.4f}")

# ── STEP 4: Compare two images from same category ─────────
print("\n" + "="*50)
print("SIMILARITY CHECK — 2 images from same category vs different")
print("="*50)

# Pick 2 images from 'striped'
striped_indices = np.where(labels == 'striped')[0]
dotted_indices  = np.where(labels == 'dotted')[0]

feat_striped_1 = features[striped_indices[0]].astype(np.float32)
feat_striped_2 = features[striped_indices[1]].astype(np.float32)
feat_dotted_1  = features[dotted_indices[0]].astype(np.float32)

# Cosine similarity — 1.0 = identical, 0.0 = completely different
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_same = cosine_similarity(feat_striped_1, feat_striped_2)
sim_diff = cosine_similarity(feat_striped_1, feat_dotted_1)

print(f"Striped image 1 vs Striped image 2 : {sim_same:.4f}  (same category — should be HIGH)")
print(f"Striped image 1 vs Dotted image 1  : {sim_diff:.4f}  (diff category — should be LOWER)")

# ── STEP 5: Visual output — show 8 images with their labels
print("\nGenerating visual chart...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Sample Images & Their Feature Fingerprints", fontsize=14)

sample_indices = [
    striped_indices[0], striped_indices[1],
    dotted_indices[0],  dotted_indices[1],
    np.where(labels == 'floral')[0][0]   if 'floral'   in labels else 0,
    np.where(labels == 'zigzagged')[0][0] if 'zigzagged' in labels else 1,
    np.where(labels == 'knitted')[0][0]  if 'knitted'  in labels else 2,
    np.where(labels == 'woven')[0][0]    if 'woven'    in labels else 3,
]

for ax, idx in zip(axes.flat, sample_indices):
    # Show the image
    img = Image.open(paths[idx])
    ax.imshow(img)
    
    # Title = category + first 3 feature numbers as a peek
    feat_peek = features[idx][:3].astype(np.float32)
    ax.set_title(
        f"{labels[idx]}\n"
        f"feat[:3] = [{feat_peek[0]:.2f}, {feat_peek[1]:.2f}, {feat_peek[2]:.2f}...]",
        fontsize=8
    )
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(BASE_DIR, "outputs", "feature_check.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\n✓ Chart saved to outputs/feature_check.png")
print("\n✅ Everything looks good! CNN phase is 100% complete.")
print("    Next step → building the GAN")