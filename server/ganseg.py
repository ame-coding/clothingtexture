import os
import uuid
from pathlib import Path

import subprocess
import argparse


parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--uuid", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
# Parse command line
args = parser.parse_args()

uid = args.uuid
pmt=args.prompt

ROOT = Path(__file__).resolve().parent.parent

subprocess.run(
    [
        "python",
        "../gan/src/main.py",
        "-cfg ../gan/src/configs/CIFAR10/BigGAN-ADA.yaml \\",
        "-ckpt ../models/model=G-current-weights-step=8000.pth \\"
        "-save ../results \\"
        "-sf \\"
        "-sf_num 10"
    ]    
)

subprocess.run(
    [
        "python",
        "../scripts/ganerate_masks.py",
        "--uuid", f"{uid}",
        "--genimg", f"{genimgpath}"
    ]    
)

subprocess.run(
    [
        "python",
        "../scripts/apply_masks.py",
        "--uuid", f"{uid}",
    ]    
)
