# CNN from Scratch (NumPy)

A Convolutional Neural Network implemented fully from first principles using NumPy.

## What’s inside
- Convolution forward pass (manual sliding-window)
- MaxPool forward pass + argmax mask
- Backpropagation through:
  - Softmax + Cross-Entropy
  - Dense layer
  - MaxPool (gradient routing)
  - ReLU
  - Convolution (dW, db, dX)
- Mini-batch training loop
- Learning curve + sample prediction visualization

## Dataset
- MNIST handwritten digit dataset (28x28 grayscale images)

## Notes
- This implementation prioritizes clarity and learning using NumPy, Linear Algebra and calculus. Full-dataset training can be slow in pure NumPy.

## How to Run

```bash
pip install numpy pandas matplotlib
python MNISTNumPyCNN.py
