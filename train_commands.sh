#!/bin/bash
# Fashion GAN Training with StudioGAN
# Direct commands to train different GAN architectures

# Set paths
PROJECT_ROOT=$(pwd)
STUDIOGAN_DIR="PyTorch-StudioGAN"
DATA_PATH="data/fashion_dataset/train"

echo "=========================================="
echo "Fashion GAN Training Setup"
echo "=========================================="

# Check if StudioGAN exists
if [ ! -d "$STUDIOGAN_DIR" ]; then
    echo "ERROR: PyTorch-StudioGAN directory not found!"
    echo "Please ensure StudioGAN is in the project root"
    exit 1
fi

cd $STUDIOGAN_DIR

# Option 1: DCGAN Training (Simple, Fast)
echo -e "\n[OPTION 1] DCGAN Training"
echo "Command:"
echo "python src/main.py \\"
echo "  -t \\"
echo "  -c src/configs/CIFAR10/DCGAN.yaml \\"
echo "  --dataset custom \\"
echo "  --data_path ../$DATA_PATH \\"
echo "  --img_size 128 \\"
echo "  --num_classes 5 \\"
echo "  --batch_size 64 \\"
echo "  --total_steps 100000 \\"
echo "  --eval_backbone InceptionV3_tf"

# Option 2: BigGAN Training (Better Quality, Conditional)
echo -e "\n[OPTION 2] BigGAN Training (RECOMMENDED)"
echo "Command:"
echo "python src/main.py \\"
echo "  -t \\"
echo "  -c src/configs/CIFAR10/BigGAN.yaml \\"
echo "  --dataset custom \\"
echo "  --data_path ../$DATA_PATH \\"
echo "  --img_size 128 \\"
echo "  --num_classes 5 \\"
echo "  --batch_size 32 \\"
echo "  --total_steps 100000 \\"
echo "  --eval_backbone InceptionV3_tf"

# Option 3: StyleGAN2 Training (Best Quality, Slow)
echo -e "\n[OPTION 3] StyleGAN2 Training (Advanced)"
echo "Command:"
echo "python src/main.py \\"
echo "  -t \\"
echo "  -c src/configs/FFHQ/StyleGAN2.yaml \\"
echo "  --dataset custom \\"
echo "  --data_path ../$DATA_PATH \\"
echo "  --img_size 128 \\"
echo "  --num_classes 5 \\"
echo "  --batch_size 16 \\"
echo "  --total_steps 150000"

echo -e "\n=========================================="
echo "To start training, copy one of the commands above"
echo "=========================================="

# Quick start - uncomment to run immediately
# echo -e "\nStarting DCGAN training..."
# python src/main.py \
#   -t \
#   -c src/configs/CIFAR10/DCGAN.yaml \
#   --dataset custom \
#   --data_path ../$DATA_PATH \
#   --img_size 128 \
#   --num_classes 5 \
#   --batch_size 64 \
#   --total_steps 100000 \
#   --eval_backbone InceptionV3_tf
