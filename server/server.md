# FastAPI Backend

This folder contains the deployment backend.

## Responsibilities

- Receive uploaded texture images.
- Receive selected garment category.
- Randomly choose a curated garment.
- Load the corresponding segmentation mask.
- Apply texture transfer.
- Return the final textured garment.

## Folder Structure

selected/
Precomputed garments.

masks/
Corresponding U²-Net masks.

uploads/
Temporary uploaded texture images.

finished/
Final generated results returned to the frontend.

## Notes

The backend does not perform GAN inference.

Garment generation and segmentation are completed offline before deployment.
