"""
texture_transfer.py

Applies a user texture onto a generated garment
using a precomputed U²-Net mask.

Author: Neel
"""

import os
import cv2
import numpy as np

TEXTURE_SCALE = 0.35

class TextureTransfer:

    def __init__(self):
        pass

    # -----------------------------------------
    # Load Images
    # -----------------------------------------

    def load(self, garment_path, mask_path, texture_path):

        garment = cv2.imread(garment_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        texture = cv2.imread(texture_path)

        if garment is None:
            raise FileNotFoundError(garment_path)

        if mask is None:
            raise FileNotFoundError(mask_path)

        if texture is None:
            raise FileNotFoundError(texture_path)

        return garment, mask, texture

    # -----------------------------------------
    # Resize or Tile Texture
    # -----------------------------------------

    def prepare_texture(self, texture, target_shape, scale=0.25):

        """
        Prepare texture while preserving its natural pattern size.

        scale:
            1.0  -> original texture size
            0.5  -> pattern becomes half as large
            0.25 -> pattern becomes four times smaller
        """

        H, W = target_shape[:2]

        # ---------------------------------------
        # Scale texture instead of forcing it
        # to garment size.
        # ---------------------------------------

        texture = cv2.resize(
            texture,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        h, w = texture.shape[:2]

        # ---------------------------------------
        # Tile if necessary
        # ---------------------------------------

        reps_y = int(np.ceil(H / h))
        reps_x = int(np.ceil(W / w))

        texture = np.tile(texture, (reps_y, reps_x, 1))

        # ---------------------------------------
        # Crop only what we need
        # ---------------------------------------

        texture = texture[:H, :W]

        return texture

    # -----------------------------------------
    # Preserve Lighting using LAB
    # -----------------------------------------

    def lighting_transfer(self, garment, texture):

        garment_lab = cv2.cvtColor(
            garment,
            cv2.COLOR_BGR2LAB
        )

        texture_lab = cv2.cvtColor(
            texture,
            cv2.COLOR_BGR2LAB
        )

        # Keep brightness from garment

        texture_lab[:, :, 0] = garment_lab[:, :, 0]

        result = cv2.cvtColor(
            texture_lab,
            cv2.COLOR_LAB2BGR
        )

        return result

    # -----------------------------------------
    # Blend using Mask
    # -----------------------------------------

    def blend(self,
              garment,
              textured,
              mask):

        mask = cv2.GaussianBlur(mask, (7,7), 0)

        alpha = mask.astype(np.float32) / 255.0

        alpha = alpha[:, :, np.newaxis]

        result = (
            textured * alpha +
            garment * (1-alpha)
        )

        return result.astype(np.uint8)

    # -----------------------------------------
    # Main Function
    # -----------------------------------------

    def apply_texture(self,
                      garment_path,
                      mask_path,
                      texture_path,
                      output_path):

        garment, mask, texture = self.load(
            garment_path,
            mask_path,
            texture_path
        )

        texture = self.prepare_texture(
            texture,
            garment.shape,
            scale=TEXTURE_SCALE
        )

        textured = self.lighting_transfer(
            garment,
            texture
        )

        final = self.blend(
            garment,
            textured,
            mask
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        cv2.imwrite(
            output_path,
            final
        )

        print("Saved:", output_path)


if __name__ == "__main__":

    tt = TextureTransfer()

    tt.apply_texture(

        garment_path="outputs/selected/shirt/shirt_01.png",

        mask_path="outputs/masks/shirt/shirt_01.png",

        texture_path="textures/banded.jpg",

        output_path="outputs/textured/shirt_01_banded.png"

    )