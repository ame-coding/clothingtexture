# scripts/evaluate_features.py (OPTIMIZED VERSION)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = r"D:\MajorProject26"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── Load saved features ───────────────────────────────────
print("Loading features...")
features = np.load(os.path.join(FEATURE_DIR, "dtd_features.npy")).astype(np.float32)
labels   = np.load(os.path.join(FEATURE_DIR, "dtd_labels.npy"))
print(f"✓ Loaded: {features.shape[0]} images, {features.shape[1]} features each\n")

# ════════════════════════════════════════════════════════════
# TEST 1 — t-SNE (OPTIMIZED)
# ════════════════════════════════════════════════════════════

print("="*55)
print("TEST 1 — t-SNE Cluster Visualization")
print("="*55)

selected_categories = [
    'striped', 'dotted', 'zigzagged', 'knitted',
    'floral', 'woven', 'braided', 'chequered',
    'spiralled', 'frilly'
]

mask = np.isin(labels, selected_categories)

# 🔥 LIMIT DATA (CRITICAL FIX)
MAX_SAMPLES = 2500
features_sub = features[mask][:MAX_SAMPLES]
labels_sub   = labels[mask][:MAX_SAMPLES]

print(f"Using {features_sub.shape[0]} images (limited for memory safety)")

# 🔥 PCA BEFORE TSNE (CRITICAL FIX)
print("Reducing dimensions using PCA (100352 → 50)...")
pca = PCA(n_components=50)
features_sub = pca.fit_transform(features_sub)
print("✓ PCA done:", features_sub.shape)

# t-SNE
print("\nRunning t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    verbose=1
)

features_2d = tsne.fit_transform(features_sub)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.get_cmap('tab10', len(selected_categories))

for i, cat in enumerate(selected_categories):
    cat_mask = labels_sub == cat
    ax.scatter(
        features_2d[cat_mask, 0],
        features_2d[cat_mask, 1],
        c=[colors(i)],
        label=cat,
        s=20,
        alpha=0.7
    )

ax.set_title("t-SNE Clusters (after PCA)")
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tsne.png"), dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════
# TEST 2 — KNN
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("TEST 2 — KNN Classification (Fixed)")
print("="*55)

# ✅ STEP 1 — LIMIT DATA
MAX_SAMPLES_KNN = 3000   # keep it safe
features_knn = features[:MAX_SAMPLES_KNN]
labels_knn   = labels[:MAX_SAMPLES_KNN]

print(f"Using {features_knn.shape[0]} samples for KNN")

# ✅ STEP 2 — PCA (VERY IMPORTANT)
from sklearn.decomposition import PCA

print("Reducing dimensions (100352 → 100)...")
pca_knn = PCA(n_components=100)
features_knn = pca_knn.fit_transform(features_knn)

print("✓ PCA done:", features_knn.shape)

# ✅ STEP 3 — Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_encoded = le.fit_transform(labels_knn)

# ✅ STEP 4 — Split (NOW SAFE)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_knn, labels_encoded,   # ✅ USE REDUCED DATA
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# ✅ STEP 5 — Train KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
knn.fit(X_train, y_train)

# ✅ STEP 6 — Evaluate
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\n✅ Accuracy: {accuracy:.2f}%")

# ════════════════════════════════════════════════════════════
# TEST 3 — SIMILARITY GAP
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("TEST 3 — Similarity Gap")
print("="*55)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

test_categories = ['striped', 'dotted', 'knitted']
sample_size = 20

gaps = []

for cat in test_categories:
    cat_idx = np.where(labels == cat)[0][:sample_size]
    other_idx = np.where(labels != cat)[0][:sample_size]

    intra = []
    inter = []

    for i in range(len(cat_idx)-1):
        intra.append(cosine_similarity(features[cat_idx[i]], features[cat_idx[i+1]]))

    for i in range(len(cat_idx)):
        inter.append(cosine_similarity(features[cat_idx[i]], features[other_idx[i]]))

    gap = np.mean(intra) - np.mean(inter)
    gaps.append(gap)

    print(f"{cat}: gap = {gap:.4f}")

avg_gap = np.mean(gaps)

# ════════════════════════════════════════════════════════════
# FINAL RESULT
# ════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("FINAL RESULT")
print("="*55)
print(f"KNN Accuracy : {accuracy:.2f}%")
print(f"Similarity Gap: {avg_gap:.4f}")

if accuracy >= 50 and avg_gap > 0:
    print("\n✅ Features are GOOD → Ready for GAN")
else:
    print("\n⚠️ Features need improvement")