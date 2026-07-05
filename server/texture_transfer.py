"""
texture_transfer.py
Poisson blending based texture transfer using precomputed U2-Net masks.
"""

import os
import cv2
import numpy as np


class TextureTransfer:

    def apply_texture(self, garment_path, mask_path, texture_path, output_path):

        # Load
        garment = cv2.imread(garment_path)
        mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        texture = cv2.imread(texture_path)

        if garment is None: raise FileNotFoundError(garment_path)
        if mask    is None: raise FileNotFoundError(mask_path)
        if texture is None: raise FileNotFoundError(texture_path)

        gh, gw = garment.shape[:2]

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Bounding box of the garment region (not fixed 128x128 assumption)
        x, y, box_w, box_h = cv2.boundingRect(mask)

        # Resize texture to the garment's actual size, then tile it to
        # cover the whole garment canvas so seamlessClone has full coverage
        tex_resized = cv2.resize(texture, (box_w, box_h))
        reps_y = gh // box_h + 2
        reps_x = gw // box_w + 2
        tiled = np.tile(tex_resized, (reps_y, reps_x, 1))[:gh, :gw]

        # Centroid of the mask — this is where seamlessClone needs its center,
        # NOT a hardcoded point
        M = cv2.moments(mask)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

        # Poisson blending
        result = cv2.seamlessClone(tiled, garment, mask, center, cv2.MIXED_CLONE)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, result)
        print("Saved:", output_path)
        return output_path


if __name__ == "__main__":
    tt = TextureTransfer()
    tt.apply_texture(
        garment_path="selected/shirt/shirt_01.png",
        mask_path="masks/shirt/shirt_01.png",
        texture_path="textures/denim.jpg",
        output_path="finished/shirt_01_denim.png"
    )