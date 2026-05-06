import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========================
# CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_PATH = os.path.join(BASE_DIR, "features", "features.dat")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "feature_compressor.pth")
OUTPUT_PATH = os.path.join(BASE_DIR, "features", "compressed_features.npy")

NUM_SAMPLES = 5640
INPUT_DIM = 100352
BATCH_SIZE = 32
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# DATASET (MEMMAP)
# ========================
class FeatureDataset(Dataset):
    def __init__(self):
        self.data = np.memmap(
            FEATURE_PATH,
            dtype="float32",
            mode="r",
            shape=(NUM_SAMPLES, INPUT_DIM)
        )

    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, idx):
        x = self.data[idx]
        return torch.tensor(x, dtype=torch.float32)

# ========================
# MODEL
# ========================
class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return self.net(x)

# ========================
# TRAIN
# ========================
dataset = FeatureDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Compressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("🚀 Training compressor...")

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        output = model(batch)

        # simple L2 loss (autoencoder-like)
        loss = (output ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ========================
# SAVE MODEL
# ========================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved: {MODEL_SAVE_PATH}")

# ========================
# GENERATE COMPRESSED FEATURES
# ========================
print("⚡ Generating compressed features...")

compressed = []

model.eval()

with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=64):
        batch = batch.to(device)
        out = model(batch)
        compressed.append(out.cpu().numpy())

compressed = np.vstack(compressed)

np.save(OUTPUT_PATH, compressed)

print("🎉 Compression complete")
print(f"Compressed shape: {compressed.shape}")