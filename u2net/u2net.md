# U²-Net

Contains the segmentation model used for garment extraction.

## Purpose

Generate binary masks for the generated garments.

## Folder Structure

model/
U²-Net implementation.

saved_models/
Pretrained model weights (ignored by Git).

## Deployment

Segmentation is performed offline.

The website only uses the precomputed masks stored in the server directory.
