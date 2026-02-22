import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(r'C:\Users\yoavn\Downloads\archive\train.csv')
data.head()

data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T # 785x1000
Y_dev = data_dev[0] # (1000,) vector
X_dev = data_dev[1:n] # 784x1000
X_dev = X_dev / 255. # normalize input 

data_test = data[1000:2000].T # 785x1000
Y_test = data_test[0] # (1000,) vector
X_test = data_test[1:n] # 784x1000
X_test = X_test / 255. # normalize input

data_train = data[2000:m].T # 785xm
Y_train = data_train[0] # (m,)
X_train = data_train[1:n] # 784xm
X_train = X_train / 255. # normalize input

def init_params(hidden_size=10):
    W1 = np.random.randn(hidden_size, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(10, hidden_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    # prevent overflow
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0


def plot_learning_curves_dense(history):
    it = history["iter"]

    # Accuracy
    plt.figure()
    plt.plot(it, history["train_acc"], label="Train Acc")
    if len(history["dev_acc"]) > 0:
        plt.plot(it, history["dev_acc"], label="Dev Acc")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Accuracy)")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(it, history["train_loss"], label="Train Loss")
    if len(history["dev_loss"]) > 0:
        plt.plot(it, history["dev_loss"], label="Dev Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()
    plt.show()

def plot_zoomed_accuracy_dense(history, low=0.85):
    it = history["iter"]
    plt.figure()
    plt.plot(it, history["train_acc"], label="Train Acc")
    if len(history["dev_acc"]) > 0:
        plt.plot(it, history["dev_acc"], label="Dev Acc")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Accuracy - Zoomed)")
    plt.ylim(low, 1.0)
    plt.legend()
    plt.show()



def evaluate_loss_acc(W1, b1, W2, b2, X_val, Y_val):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_val)
    preds = get_predictions(A2)
    acc = get_accuracy(preds, Y_val)

    one_hot_Y = one_hot(Y_val)
    loss = compute_loss(A2, one_hot_Y)
    return loss, acc

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, one_hot_Y, lambd):
    m = X.shape[1]
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T) + (lambd / m) * W2 # regularisation imp (ridge regression)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True) # shape: (10,1)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T) + (lambd / m) * W1 # regularisation imp (ridge regression)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) 
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2  
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compute_loss(A2, one_hot_Y, eps=1e-12):
    return -np.mean(np.sum(one_hot_Y * np.log(A2 + eps), axis=0))

def gradient_descent(X, Y, alpha, iterations, lambd=0.01, hidden_size=10,
                     X_dev=None, Y_dev=None, log_every=10):
    W1, b1, W2, b2 = init_params(hidden_size)
    one_hot_Y = one_hot(Y)

    history = {
        "iter": [],
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, one_hot_Y, lambd)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % log_every == 0 or i == iterations - 1:
            train_loss = compute_loss(A2, one_hot_Y)
            preds = get_predictions(A2)
            train_acc = get_accuracy(preds, Y)

            history["iter"].append(i)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_dev is not None and Y_dev is not None:
                dev_loss, dev_acc = evaluate_loss_acc(W1, b1, W2, b2, X_dev, Y_dev)
                history["dev_loss"].append(dev_loss)
                history["dev_acc"].append(dev_acc)
                print(f"Iter {i}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}")
            else:
                print(f"Iter {i}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    return W1, b1, W2, b2, history

W1, b1, W2, b2, history = gradient_descent( X_train, Y_train,alpha=0.05, iterations=501,lambd=0.0, hidden_size=10, X_dev=X_dev, Y_dev=Y_dev,log_every=10)

plot_learning_curves_dense(history)
plot_zoomed_accuracy_dense(history, low=0.85)


def evaluate_model(W1, b1, W2, b2, X_val, Y_val):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_val)
    predictions = get_predictions(A2)
    print("Predictions:", predictions[:10])
    print("Y_test     :", Y_val[:10])
    print("Shape of predictions:", predictions.shape)
    print("Shape of Y_test     :", Y_val.shape)
    return get_accuracy(predictions, Y_val)

dev_accuracy = evaluate_model(W1, b1, W2, b2, X_dev, Y_dev)
test_accuracy = evaluate_model(W1, b1, W2, b2, X_test, Y_test)
print("Dev Accuracy:", dev_accuracy)
print("Test Accuracy:", test_accuracy)

def find_best_hidden_size(X_train, Y_train, X_val, Y_val, alpha, iterations, lambd, hidden_sizes):
    best_size = None
    best_accuracy = 0
    results = []

    for size in hidden_sizes:
        print(f"\nTraining with hidden layer size = {size}")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations, lambd, hidden_size=size)
        acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
        results.append((size, acc))
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_size = size

    print(f"\n Best hidden size: {best_size} with accuracy: {best_accuracy:.4f}")
    return best_size, results

hidden_sizes = [128, 110]
#best_size, results = find_best_hidden_size(X_train, Y_train, X_dev, Y_dev, alpha=0.1, iterations=500, lambd=0, hidden_sizes=hidden_sizes)

def evaluate_model(W1, b1, W2, b2, X_val, Y_val):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_val)
    predictions = get_predictions(A2)
    print("Predictions:", predictions[:10])
    print("Y_test     :", Y_val[:10])
    print("Shape of predictions:", predictions.shape)
    print("Shape of Y_test     :", Y_val.shape)
    return get_accuracy(predictions, Y_val)


def find_best_lambda(X_train, Y_train, X_val, Y_val, alpha, iterations, lambdas, hidden_size):
    best_lambda = None
    best_accuracy = 0
    results = []

    for lambd in lambdas:
        print(f"Training with lambda = {lambd}")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations, lambd, hidden_size)
        acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
        results.append((lambd, acc))
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_lambda = lambd

    print(f"\n Best lambda: {best_lambda} with accuracy: {best_accuracy:.4f}")
    return best_lambda, results

#lambdas = [0.01, 0.1, 1, 2]
#best_lambda, results = find_best_lambda(X_train, Y_train, X_test, Y_test, 0.1, 501, lambdas, 110)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index].reshape(-1, 1)   # Transposing
    prediction = make_predictions(current_image, W1, b1, W2, b2)[0]
    label = Y_train[index]

    print("Prediction:", prediction)
    print("Label     :", label)

    img = current_image.reshape(28, 28) # Normalizing image to 28x28 for visualization
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

for i in range(5, 1000, 100):
    test_prediction(i, W1, b1, W2, b2)
