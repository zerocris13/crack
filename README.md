# Crack Segmentation with CrackFormer-II

This repository contains an implementation of **CrackFormer-II** for automatic crack segmentation using deep learning. The objective is to identify cracks at the pixel level from RGB images and generate a binary segmentation mask separating **crack** and **background** regions.

## Dataset

The experiments were performed using the **DeepCrack dataset**, a public benchmark for crack segmentation.

The dataset contains:

* **537 RGB images** with manually annotated segmentation masks.
* **300 images** for training.
* **237 images** for testing.
* Binary pixel-level annotations representing crack and background regions.

DeepCrack includes cracks with different shapes, widths, scales and surface conditions, making it useful for evaluating crack segmentation models.

## Methodology

The general workflow of the project is:

1. **Data preparation:** RGB images are paired with their corresponding binary segmentation masks.
2. **Preprocessing:** Images and masks are prepared and normalized before being provided to the network.
3. **Model:** **CrackFormer-II** is used as the semantic segmentation architecture. The model is based on self-attention mechanisms designed to capture both fine crack details and contextual information.
4. **Training:** The network learns to classify each pixel as crack or background using supervised learning.
5. **Inference:** The trained model generates a probability map that is converted into a binary crack mask using a segmentation threshold.
6. **Evaluation:** Segmentation performance can be evaluated using metrics such as **IoU, F1/Dice Score, Precision and Recall**.

## Repository Structure

```text
crack/
│
├── app.py
├── crackformerII.py
├── lossFunctions.py
├── requirements.txt
└── models/
```

* `crackformerII.py`: CrackFormer-II model architecture.
* `lossFunctions.py`: loss functions used for segmentation.
* `models/`: trained model weights.
* `app.py`: application/inference pipeline.
* `requirements.txt`: Python dependencies.

## Reference

This project is based on the CrackFormer-II architecture presented in:

**Liu, H., Yang, J., Miao, X., Mertz, C., & Kong, H. (2023).**
*CrackFormer Network for Pavement Crack Segmentation.*
IEEE Transactions on Intelligent Transportation Systems, 24(9), 9240–9252.
https://doi.org/10.1109/TITS.2023.3266776

Original implementation:
https://github.com/LouisNUST/CrackFormer-II
