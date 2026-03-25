# MNIST Dense Neural Network from Scratch (NumPy)

This project implements a fully connected neural network for handwritten digit classification from first principles using **NumPy**.

Rather than relying on deep learning frameworks, the model was built manually to better understand how core neural network components work internally — including forward propagation, backpropagation, activation functions, loss calculation, and parameter updates.

The project also goes beyond a single baseline model by comparing multiple hidden-layer sizes and analyzing the tradeoff between **predictive performance** and **computational efficiency**.

---

## Features

- Manual forward propagation
- Manual backpropagation with gradient computation
- ReLU activation
- Softmax + Cross-Entropy loss
- L2 regularization
- Hidden-layer size comparison (`10`, `64`, `128`, `256`)
- Training / validation / test evaluation
- Learning curve visualization
- Baseline vs best-model overlay plots
- Runtime benchmarking
- Parameter count analysis
- Weighted efficiency scoring
- Log-based efficiency scoring
- Relative improvement analysis versus baseline

---

## Dataset

- **MNIST handwritten digit dataset**
- 28x28 grayscale images
- Multi-class classification problem (digits `0-9`)

---

## Experiment Design

The project begins with a baseline dense neural network and then extends into a hidden-layer sweep to compare different model sizes.

### Evaluated hidden sizes
- `10` (baseline)
- `64`
- `128`
- `256`

### Evaluation workflow
- Train each model on the training set
- Track training and validation performance during training
- Select models based on **validation performance**
- Use the test set only for final comparison
- Compare architectures not only by accuracy, but also by:
  - runtime
  - parameter count
  - efficiency metrics
  - relative improvement over baseline

This allows the project to be analyzed both as a machine learning implementation and as a small optimization / model selection experiment.

---

## Core Methodology

The neural network is implemented manually in NumPy and includes:

- Input layer: `784` features (flattened 28x28 image)
- One hidden layer (size varies by experiment)
- ReLU activation in the hidden layer
- Softmax output layer for 10-class classification
- Cross-Entropy loss
- L2 regularization
- Gradient descent parameter updates

The goal of the project was not only to classify MNIST effectively, but also to understand how architectural choices affect:

- generalization performance
- convergence behavior
- computational cost
- model efficiency

---

## Results & Findings

The hidden-layer comparison showed a clear tradeoff between model complexity and performance.

### Main findings

- Increasing hidden size from `10` to `64` produced the largest improvement in validation accuracy
- Larger models (`128`, `256`) increased parameter count significantly
- Beyond `64`, performance gains became smaller and less consistent on validation data
- The `64`-unit model emerged as the best **practical tradeoff** in the tested range

### Efficiency interpretation

Two different efficiency perspectives were used:

- **Weighted normalized efficiency score**  
  Balances validation accuracy, runtime, and parameter count  
  → Best model: **hidden size = 64**

- **Log-based efficiency score**  
  Uses a stricter “accuracy per computational cost” style metric  
  → Best model: **hidden size = 10**

This highlights an important conclusion:

> The best architecture depends on the objective.  
> If the goal is the best balanced tradeoff between accuracy and cost, `64` is the strongest choice.  
> If the goal is strict computational frugality, the baseline model remains highly competitive.

---

## Performance Summary

### Baseline model (`hidden_size = 10`)
- Training Accuracy: **89.3%**
- Validation Accuracy: **86.8%**
- Test Accuracy: **89.9%**

### Best practical tradeoff (`hidden_size = 64`)
- Highest validation accuracy among tested models
- Best weighted efficiency score
- Faster convergence and lower loss than baseline
- Strong improvement without the excessive complexity growth of larger models

---

## Visualizations

### Baseline Accuracy Learning Curve
![Baseline Accuracy Curve](images/Dense_NN1.png)

This plot shows how the baseline model (`hidden_size = 10`) improves over training iterations on both the training and validation sets.

---

### Baseline Loss Learning Curve
![Baseline Loss Curve](images/Dense_NN2.png)

The baseline loss decreases steadily on both training and validation sets, showing stable optimization behavior without major divergence.

---

### Test Accuracy vs Hidden Size
![Test Accuracy vs Hidden Size](images/Dense_NN4.png)

This comparison shows how final test accuracy changes as hidden-layer size increases. Accuracy improves significantly from the baseline and then begins to plateau, indicating diminishing returns for larger architectures.

---

### Weighted Efficiency Score vs Hidden Size
![Efficiency Score vs Hidden Size](images/Dense_NN5.png)

This metric combines validation accuracy, runtime, and parameter count into a balanced efficiency score. The `64`-unit model achieves the best overall tradeoff in the tested range.

---

### Accuracy Delta vs Baseline
![Accuracy Delta vs Baseline](images/Dense_NN6.png)

This plot compares how much each model improves over the baseline (`hidden_size = 10`) in both validation and test accuracy. The largest validation gain appears at `64`, while larger models offer only marginal additional improvement.

---

### Parameter Ratio vs Baseline
![Parameter Ratio vs Baseline](images/Dense_NN7.png)

This figure highlights how model complexity grows relative to the baseline. While performance gains begin to level off, parameter count increases sharply, reinforcing the diminishing-returns pattern.

---

### Learning Curve Overlay: Baseline vs Best Accuracy Model
![Learning Curve Overlay](images/Dense_NN8.png)

This overlay compares the baseline model (`10`) with the best validation-accuracy model (`64`). The `64`-unit model converges faster, reaches higher accuracy, and maintains a stronger validation trajectory throughout training.

---

## Key Takeaway

This project demonstrates that even in a simple dense neural network, **bigger is not always better**.

A larger hidden layer improves performance only up to a point. In this experiment:

- `64` hidden units provided the strongest balance between accuracy and computational cost
- `128` and `256` increased complexity substantially
- The additional cost of larger models did not produce proportional validation gains

This turns the project from a basic MNIST classifier into a more meaningful **architecture selection and efficiency tradeoff study**.

---

## How to Run

```bash
pip install numpy pandas matplotlib
python MNISTtrainingNN.py
