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
        "-sf_num 1"
    ]    
)

results_dir = ROOT /"results"

for file in results_dir.iterdir():
    if (
        file.is_file()
        and not file.name.startswith("clothinggen_")
        and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ):
        file.rename(results_dir / f"clothinggen_{uid}.jpg")
        break

subprocess.run(
    [
        "python",
        "../scripts/ganerate_masks.py",
        "--uuid", f"{uid}",

    ]    
)

subprocess.run(
    [
        "python",
        "../scripts/apply_masks.py",
        "--uuid", f"{uid}",
    ]    
)
