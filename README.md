# Deep Learning Vision Models

A collection of deep learning implementations built from scratch and with PyTorch, focusing on image classification tasks.

This repository demonstrates progression from first-principles neural networks to production-ready CNN architectures.

---

## Projects Overview

| Model | Framework | Task | Test Accuracy |
|-------|----------|------|---------------|
| Dense Neural Network | NumPy | MNIST Digits | 89.9% |
| CNN from Scratch | NumPy | MNIST Digits | ~95% |
| CNN | PyTorch | MNIST Digits | ~98% |
| CNN | PyTorch | Sign Language MNIST | ~98–99% |

---
## Repository Structure

```text
deep-learning-vision-models/
│
├── dense-from-scratch/
│   ├── MNISTtrainingNN.py
│   ├── README.md
│   └── images/
│
├── cnn-from-scratch-numpy/
│   ├── MNISTNumPyCNN.py
│   ├── README.md
│   └── images/
│
├── mnist-pytorch/
│   ├── MNISTPyTorchCNN.py
│   ├── README.md
│   └── images/
│
└── sign-language-pytorch/
    ├── MNISTSignLanguagePyTorchCNN.py
    ├── README.md
    └── images/
