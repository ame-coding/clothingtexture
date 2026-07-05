#!/usr/bin/env python3
"""
Pre-training checklist for Fashion GAN
Verifies dataset, StudioGAN, and environment setup
"""

import os
from pathlib import Path
import sys

def check_item(name, condition, details=""):
    """Print colored check/cross for verification items"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if details and condition:
        print(f"  → {details}")
    elif details and not condition:
        print(f"  → ERROR: {details}")
    return condition

def main():
    print("\n" + "="*70)
    print("Fashion GAN Pre-Training Checklist")
    print("="*70 + "\n")
    
    all_good = True
    
    # 1. Check StudioGAN
    print("1. StudioGAN Setup")
    studiogan = Path("PyTorch-StudioGAN")
    studiogan_src = studiogan / "src"
    studiogan_main = studiogan_src / "main.py"
    
    all_good &= check_item(
        "StudioGAN directory exists",
        studiogan.exists(),
        str(studiogan.absolute()) if studiogan.exists() else "Not found"
    )
    all_good &= check_item(
        "StudioGAN src directory exists",
        studiogan_src.exists(),
        str(studiogan_src) if studiogan_src.exists() else "Missing src/"
    )
    all_good &= check_item(
        "StudioGAN main.py exists",
        studiogan_main.exists(),
        str(studiogan_main) if studiogan_main.exists() else "Missing main.py"
    )
    
    # 2. Check Dataset
    print("\n2. Dataset Structure")
    data_path = Path("data/fashion_dataset/train")
    categories = ["blazer", "dress", "pants", "shirt", "skirt"]
    
    all_good &= check_item(
        "Training data directory exists",
        data_path.exists(),
        str(data_path.absolute()) if data_path.exists() else "Not found"
    )
    
    if data_path.exists():
        for cat in categories:
            cat_path = data_path / cat
            if cat_path.exists():
                num_images = len(list(cat_path.glob("*.*")))
                check_item(
                    f"Category '{cat}' exists",
                    True,
                    f"{num_images} images"
                )
            else:
                all_good &= check_item(
                    f"Category '{cat}' exists",
                    False,
                    "Missing category"
                )
    
    # 3. Check Python packages
    print("\n3. Python Environment")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
    }
    
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            check_item(f"{name} installed", True, f"import {pkg} works")
        except ImportError:
            all_good &= check_item(f"{name} installed", False, f"pip install {pkg}")
    
    # 4. Check CUDA
    print("\n4. GPU Setup")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        check_item(
            "CUDA available",
            cuda_available,
            f"{torch.cuda.device_count()} GPU(s) found" if cuda_available else "No GPU detected"
        )
        if cuda_available:
            check_item(
                "GPU device",
                True,
                f"{torch.cuda.get_device_name(0)}"
            )
    except:
        all_good &= check_item("CUDA check", False, "PyTorch not installed")
    
    # 5. Check output directories
    print("\n5. Output Directories")
    output_dirs = [
        "results/checkpoints",
        "results/samples",
        "results/logs"
    ]
    
    for dir_path in output_dirs:
        p = Path(dir_path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            check_item(f"Created {dir_path}", True, str(p.absolute()))
        else:
            check_item(f"{dir_path} exists", True, str(p.absolute()))
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✓ All checks passed! Ready to start training.")
        print("\nQuick start commands:")
        print("  python train_fashion.py --model dcgan")
        print("  python train_fashion.py --model biggan")
    else:
        print("✗ Some checks failed. Please fix the issues above before training.")
    print("="*70 + "\n")
    
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
