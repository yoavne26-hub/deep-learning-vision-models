# Deep Learning Vision Models

A collection of deep learning implementations built from scratch and with PyTorch, focusing on image classification tasks.

This repository demonstrates progression from first-principles neural networks to production-ready CNN architectures.
---

## Technical Summary

This repository documents a structured progression through deep learning system design, starting from first-principles neural network implementation to advanced convolutional architectures in PyTorch.

Core competencies demonstrated:

- Manual implementation of forward and backward propagation
- Gradient computation without automatic differentiation
- Convolutional neural network design from scratch
- Training optimization using Adam and AdamW
- Regularization techniques (Dropout, L2 weight decay, BatchNorm)
- Training stability analysis via learning curves
- Multi-class evaluation using confusion matrices
- Generalization control through data augmentation

The progression reflects a deliberate transition from mathematical foundations (NumPy-based implementation) to scalable deep learning pipelines using modern frameworks.

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
