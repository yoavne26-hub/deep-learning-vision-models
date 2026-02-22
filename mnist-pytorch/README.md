# MNIST CNN with PyTorch

This project implements a Convolutional Neural Network for MNIST digit classification using PyTorch.

---

## Architecture

Conv → ReLU → Conv → ReLU → MaxPool → Flatten → Fully Connected → Softmax

- 2 Convolutional layers
- MaxPooling
- Adam optimizer
- CrossEntropyLoss
- GPU support (if available)

---

## Dataset

- MNIST handwritten digit dataset (28x28 grayscale images)

---

## Features
- GPU acceleration (CUDA support)
  
- Mini-batch training
  
- Train & validation tracking
  
- Learning curve visualization
  
- Single image prediction testing

---

## Educational Goal
- This implementation demonstrates the transition from manual NumPy backpropagation to a framework-based deep learning pipeline.


---

## Results

### Learning Curve (Accuracy X Epoch)
![Accuracy Curve For Epoch](images/CNN_Digit3.png)

### Learning Curve (Loss X Epoch)
![Loss Curve For Epoch](images/CNN_Digit4.png)

### Learning Curve (Accuracy X Batch)
![Accuracy Curve For Batch](images/CNN_Digit2.png)

### Learning Curve (Loss X Batch)
![Loss Curve For Batch](images/CNN_Digit1.png)

### Example
![Example](images/CNN_Digit5.png)

---

## Performance

- Train Accuracy: XX.X%
- Validation Accuracy: XX.X%
- Test Accuracy: XX.X%

---

## How to Run

```bash
pip install torch torchvision numpy pandas matplotlib
python MNISTPyTorchCNN.py
