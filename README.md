# 1Bit-Audio-KWS

# 1-Bit ResNet-18 for Speech Commands 🎙️

This repository contains a PyTorch implementation of a 1-Bit ResNet-18 trained on the Google Speech Commands (v0.02) dataset. 

To stabilize the complex binarized training process, the model is trained using **Knowledge Distillation** from a full-precision teacher.

## 📊 Results

| Model | Weight Precision | Activation Precision | Accuracy | Weight |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet18 FP32** | FP32 | FP32 | **94.8%** | 42.8 MB |
| **ResNet18 1-Bit** | 1-Bit | 1-Bit | **91.9%** | 2.2 MB |

*(Note: In accordance with BNN industry standards, the very first convolutional layer and the final classification layer are kept in FP32).*

## TODO
- Implement CUDA kernels for 1bit multiplicaition
- Compare inference speed of patched, base 1Bit model and full precision one.
