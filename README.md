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
| Dense Neural Network | NumPy | MNIST Digits | 86.7% |
| CNN from Scratch | NumPy | MNIST Digits | NA |
| CNN | PyTorch | MNIST Digits | ~98% |
| CNN | PyTorch | Sign Language MNIST | ~98–99% |


---

## Methodology

Each project in this repository follows a structured experimental workflow:

### 1. Data Preparation
- Dataset loading and normalization
- Label remapping where required
- Train / validation / test splitting
- Data augmentation (for CNN models)

### 2. Model Design
- Architecture selection based on task complexity
- Progressive increase in model depth
- Use of convolutional layers for spatial feature extraction
- Application of regularization (Dropout, BatchNorm, L2)

### 3. Optimization Strategy
- CrossEntropyLoss for multi-class classification
- Adam / AdamW optimizers
- Weight decay for generalization
- Monitoring both accuracy and loss per epoch

### 4. Evaluation
- Learning curve visualization
- Confusion matrix analysis
- Detection of overfitting behavior
- Class-wise performance inspection

This structured pipeline was consistently applied across all implementations to ensure fair comparison and reproducibility.

---

## Mathematical Foundations

The early stages of this repository focus on understanding neural networks at a mathematical level.

Key theoretical components implemented manually:

- Linear transformations:  
  $ \( Z = W X + b \) $

- Activation functions (ReLU, Softmax)

- Cross-Entropy Loss:  
 $ \( L = - \sum y \log(\hat{y}) \) $

- Backpropagation using chain rule differentiation

- Gradient descent updates:  
 $ \( W := W - \alpha \frac{\partial L}{\partial W} \) $

The NumPy implementations avoid automatic differentiation entirely, reinforcing a deep understanding of gradient flow, tensor shapes, and optimization mechanics before transitioning to PyTorch.

This progression ensures conceptual mastery rather than framework dependency.

---

## Personal Note

This entire repository was built through self-directed learning.

The goal was not only to achieve high accuracy, but to deeply understand how neural networks operate — from mathematical derivation to practical implementation.

If you have suggestions, improvements, or technical feedback, I would genuinely appreciate hearing them. Constructive input is always welcome.

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

---

