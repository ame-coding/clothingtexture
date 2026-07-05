# Fashion GAN Training Guide
## Training GANs on Masked Fashion Dataset with StudioGAN

---

## Quick Start

### 1. Check Your Setup
```bash
python check_setup.py
```
This verifies:
- StudioGAN installation
- Dataset structure  
- Python packages
- GPU availability

### 2. Start Training

**Option A: DCGAN (Recommended for beginners)**
```bash
python train_fashion.py --model dcgan --img-size 128 --steps 100000
```

**Option B: BigGAN (Better quality, uses class labels)**
```bash
python train_fashion.py --model biggan --img-size 128 --steps 100000
```

**Option C: StyleGAN2 (Best quality, slower)**
```bash
python train_fashion.py --model stylegan2 --img-size 128 --steps 150000
```

---

## What Each GAN Does

### DCGAN (Deep Convolutional GAN)
- **Best for**: Getting started quickly
- **Speed**: Fast training (~1-2 days on GPU)
- **Quality**: Good baseline results
- **Use when**: You want to validate your pipeline works

### BigGAN  
- **Best for**: High-quality conditional generation
- **Speed**: Moderate (~2-3 days on GPU)
- **Quality**: Much better than DCGAN
- **Special feature**: Can generate specific classes (blazer, dress, etc.)
- **Use when**: You want good results with class control

### StyleGAN2
- **Best for**: State-of-the-art quality
- **Speed**: Slow (~5-7 days on GPU)
- **Quality**: Best possible
- **Use when**: You need publication-quality outputs

---

## Dataset Requirements

Your masked fashion dataset should be structured like:
```
data/fashion_dataset/train/
├── blazer/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── dress/
│   ├── image_001.png
│   └── ...
├── pants/
├── shirt/
└── skirt/
```

**Image requirements**:
- Format: PNG, JPG, or JPEG
- Size: All images should be same size (64x64, 128x128, or 256x256)
- Masked: Background removed (transparent or white)
- Minimum: ~500 images per category (more is better)

---

## Training Process

### What Happens During Training

1. **Initialization** (Step 0-1000)
   - Generator creates random noise
   - Discriminator learns basic features
   - Images look like random patterns

2. **Early Training** (Step 1000-10000)
   - Basic shapes emerge
   - Colors start to make sense
   - Still blurry and incomplete

3. **Mid Training** (Step 10000-50000)
   - Recognizable clothing items
   - Better textures and details
   - Some artifacts may appear

4. **Late Training** (Step 50000-100000)
   - High-quality fashion items
   - Sharp details and realistic textures
   - Class-specific features clear

### Monitoring Training

**Checkpoints saved every 5000 steps to**:
```
results/checkpoints/custom-fashion_<MODEL>-train-<TIMESTAMP>/
```

**Sample images saved to**:
```
results/samples/
```

**Training logs**:
```
results/logs/
```

### Stopping and Resuming

**Stop training**: Press `Ctrl+C`

**Resume from checkpoint**:
```bash
python train_fashion.py --model dcgan --resume results/checkpoints/<checkpoint_folder>
```

---

## After Training: Generating New Fashion Items

### Generate Random Samples
```python
import torch
from PyTorch-StudioGAN.src import utils

# Load trained generator
checkpoint_path = "results/checkpoints/custom-fashion_DCGAN-train-*/model_best.pth"
generator = load_generator(checkpoint_path)

# Generate random fashion items
z = torch.randn(16, 128)  # 16 random latent vectors
class_labels = torch.randint(0, 5, (16,))  # Random classes

generated_images = generator(z, class_labels)
save_images(generated_images, "outputs/generated_fashion.png")
```

### Generate Specific Class
```python
# Generate only dresses (class index 1)
z = torch.randn(10, 128)
class_labels = torch.ones(10, dtype=torch.long) * 1  # All dresses

dresses = generator(z, class_labels)
```

### Class Indices
- 0: Blazer
- 1: Dress  
- 2: Pants
- 3: Shirt
- 4: Skirt

---

## Troubleshooting

### "Out of memory" error
- Reduce `--batch-size` (try 16 or 32)
- Reduce `--img-size` (try 64 instead of 128)
- Use DCGAN instead of BigGAN/StyleGAN2

### Training is too slow
- Use smaller `--img-size` (64 instead of 128)
- Use DCGAN instead of BigGAN
- Reduce `--steps` for faster experimentation

### Poor quality results
- Train for more steps (100000+)
- Use larger `--img-size` (128 or 256)
- Try BigGAN instead of DCGAN
- Ensure dataset has enough variety (500+ images per class)

### Mode collapse (all outputs look similar)
- This is common in GANs
- Try BigGAN with hinge loss
- Reduce discriminator learning rate
- Add data augmentation

---

## Expected Training Times (on RTX 3090)

| Model      | Image Size | Batch Size | Steps   | Time     |
|------------|------------|------------|---------|----------|
| DCGAN      | 128x128    | 64         | 100K    | ~24 hrs  |
| BigGAN     | 128x128    | 32         | 100K    | ~48 hrs  |
| StyleGAN2  | 128x128    | 16         | 150K    | ~120 hrs |

---

## Next Steps After Training

1. **Evaluate quality**: Check FID score and sample diversity
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Generate dataset**: Create large batch of synthetic fashion items
4. **Apply to downstream tasks**: Use generated data for training classifiers, recommender systems, etc.

---

## Advanced: Custom Configuration

Edit config files in `PyTorch-StudioGAN/src/configs/` for:
- Learning rates
- Loss functions
- Architecture details
- Augmentation strategies

---

## Questions?

Check StudioGAN documentation: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
