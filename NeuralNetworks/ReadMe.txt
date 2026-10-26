# Contrastive Wide ResNet-50 Feature Extractor

This repository contains the infrastructure to load and utilize a contrastively trained model. The architecture is based on **Wide ResNet-50-2**, to produce high-quality, normalized image embeddings.

## 📂 Project Structure

* **wrn50_supcon.pth**: The saved model checkpoint containing the trained backbone weights.
* **Defining_LoadingTheSupConTrained.py**: The Python script responsible for defining the model architecture and loading the checkpoint weights.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following libraries installed:
pip install torch torchvision pillow