"""
generator.py

Loads the trained BigGAN generator from StudioGAN
and generates garment images.

Author: Neel
"""

import os
import sys
import types

import torch
import torch.nn as nn

from torchvision.utils import save_image


###############################################################
# PATHS
###############################################################

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STUDIOGAN_DIR = os.path.join(PROJECT_ROOT, "StudioGAN_working")

SRC_DIR = os.path.join(STUDIOGAN_DIR, "src")

UTILS_DIR = os.path.join(SRC_DIR, "utils")

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "model=G_ema-best-weights-step=22000.pth"
)


###############################################################
# Add StudioGAN to Python path
###############################################################

sys.path.append(SRC_DIR)
sys.path.append(UTILS_DIR)


###############################################################
# StudioGAN Imports
###############################################################

import ops

from models.big_resnet import Generator


###############################################################
# DEVICE
###############################################################

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("-------------------------------------")
print("Using device:", DEVICE)
print("-------------------------------------")


###############################################################
# MODULES
#
# Exactly the same as your notebook.
###############################################################

MODULES = types.SimpleNamespace()

# Generator

MODULES.g_conv2d = ops.snconv2d
MODULES.g_deconv2d = ops.sndeconv2d
MODULES.g_linear = ops.snlinear
MODULES.g_embedding = ops.sn_embedding
MODULES.g_bn = ops.ConditionalBatchNorm2d
MODULES.g_act_fn = nn.ReLU(inplace=True)

# Discriminator

MODULES.d_conv2d = ops.snconv2d
MODULES.d_deconv2d = ops.sndeconv2d
MODULES.d_linear = ops.snlinear
MODULES.d_embedding = ops.sn_embedding


###############################################################
# Dummy MODEL
#
# StudioGAN expects a MODEL object even though
# we only need inference.
###############################################################

class DummyMODEL:

    info_type = "N/A"

    g_info_injection = "concat"

    info_num_discrete_c = 0
    info_dim_discrete_c = 0
    info_num_conti_c = 0


MODEL = DummyMODEL()


###############################################################
# FashionGenerator Class
###############################################################

class FashionGenerator:

    def __init__(self):

        self.class_map = {

            "blazer": 0,

            "dress": 1,

            "pants": 2,

            "shirt": 3,

            "skirt": 4

        }

        self.generator = None

        self.load_model()
    ###########################################################
    # Load trained Generator
    ###########################################################

    def load_model(self):

        print("\nLoading Generator...\n")

        self.generator = Generator(

            z_dim=120,

            g_shared_dim=128,

            img_size=128,

            g_conv_dim=32,

            apply_attn=True,

            attn_g_loc=[2],

            g_cond_mtd="cBN",

            num_classes=5,

            g_init="ortho",

            g_depth=1,

            mixed_precision=False,

            MODULES=MODULES,

            MODEL=MODEL

        ).to(DEVICE)

        #######################################################
        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
            )
        
        checkpoint = torch.load(
             CHECKPOINT,
             map_location=DEVICE,
            weights_only=False
        )

        self.generator.load_state_dict(

            checkpoint["state_dict"]

        )

        self.generator.eval()

        print("Generator loaded successfully!")

        print()

    ###########################################################
    # Generate Images
    ###########################################################

    def generate_images(
        self,
        class_name,
        num_images=20
    ):

        if class_name not in self.class_map:
            raise ValueError(
                f"Unknown class: {class_name}"
            )

        class_id = self.class_map[class_name]

        output_dir = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "generated",
            class_name
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        saved_paths = []

        print("-------------------------------------")
        print(f"Generating {num_images} {class_name} images")
        print("-------------------------------------")

        with torch.no_grad():

            for i in range(num_images):

                ##################################################
                # Random latent vector
                ##################################################

                z = torch.randn(
                    1,
                    120,
                    device=DEVICE
                )

                ##################################################
                # Class label
                ##################################################

                y = torch.LongTensor(
                    [class_id]
                ).to(DEVICE)

                ##################################################
                # Forward pass
                ##################################################

                fake = self.generator(
                    z,
                    y
                )

                ##################################################
                # Convert [-1,1] → [0,1]
                ##################################################

                fake = (fake + 1) / 2

                ##################################################
                # Save image
                ##################################################

                filename = os.path.join(
                    output_dir,
                    f"{i:03d}.png"
                )

                save_image(
                    fake,
                    filename
                )

                saved_paths.append(
                    filename
                )

        print(f"Saved {len(saved_paths)} images.\n")

        return saved_paths


###############################################################
# Main
###############################################################

if __name__ == "__main__":

    generator = FashionGenerator()

    for cls in [
        "blazer",
        "dress",
        "pants",
        "shirt",
        "skirt"
    ]:

        generator.generate_images(
            class_name=cls,
            num_images=50
        )

    # print("\nGenerated Files:\n")

    # for path in paths:
    #     print(path)