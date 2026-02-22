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
X_dev = X_dev.astype(np.float32) / 255.0 # normalizing input
X_dev_cnn = X_dev.T.reshape(-1, 1, 28, 28) # Reshape for CNN input (28x28 images with 1 channel)

data_test = data[1000:2000].T # 785x1000
Y_test = data_test[0] # (1000,) vector
X_test = data_test[1:n] # 784x1000
X_test = X_test.astype(np.float32) / 255.0 # normalizing input
X_test_cnn = X_test.T.reshape(-1, 1, 28, 28) # Reshape for CNN input (28x28 images with 1 channel)

data_train = data[2000:m].T # 785xm
Y_train = data_train[0] # (m,)
X_train = data_train[1:n] # 784xm
X_train = X_train.astype(np.float32) / 255.0 # normalizing input
X_train_cnn = X_train.T.reshape(-1, 1, 28, 28) # Reshape for CNN input (28x28 images with 1 channel)

def init_params_cnn(num_filters=8, kernel_size=3):
    C_in = 1
    fan_in = C_in * kernel_size * kernel_size  # 9 for 3x3
    Wc = np.random.randn(num_filters, C_in, kernel_size, kernel_size) * np.sqrt(2 / fan_in)
    bc = np.zeros((num_filters,))  # (F,)

    flat_dim = num_filters * 14 * 14  # after 2x2 pooling
    Wd = np.random.randn(10, flat_dim) * np.sqrt(2 / flat_dim)
    bd = np.zeros((10,)) # (10,)

    return Wc, bc, Wd, bd

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return (Z > 0).astype(Z.dtype)

def softmax(Z):
    # prevent overflow
    Z_shift = Z - np.max(Z, axis = 1, keepdims = True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)

# CNN forward prop functions

def conv_forward(X, W, b, stride=1, pad=1):
    """
    X: (m, C_in, H, W)
    W: (F, C_in, K, K)
    b: (F,)
    returns Z: (m, F, H_out, W_out)
    """
    m, C_in, H, W_in = X.shape
    F, Cw, K, Kw = W.shape
    assert Cw == C_in and K == Kw, "Kernel shape mismatch"

    H_out = (H - K + 2*pad) // stride + 1
    W_out = (W_in - K + 2*pad) // stride + 1

    X_pad = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), mode="constant")
    Z = np.zeros((m, F, H_out, W_out), dtype=X.dtype)

    for i in range(m):
        for f in range(F):
            for h in range(H_out):
                hs = h * stride
                he = hs + K
                for w in range(W_out):
                    ws = w * stride
                    we = ws + K
                    region = X_pad[i, :, hs:he, ws:we]          # (C_in, K, K)
                    Z[i, f, h, w] = np.sum(region * W[f]) + b[f]
    return Z

def maxpool_forward(X, size=2, stride=2):
    m, C, H, W = X.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1

    P = np.zeros((m, C, H_out, W_out), dtype=X.dtype)
    mask = np.zeros_like(X, dtype=bool)  # marks where max came from

    for i in range(m):
        for c in range(C):
            for h in range(H_out):
                hs = h * stride
                he = hs + size
                for w in range(W_out):
                    ws = w * stride
                    we = ws + size

                    region = X[i, c, hs:he, ws:we]  # (size,size)
                    max_val = np.max(region)
                    P[i, c, h, w] = max_val

                    # mark argmax location (if ties, this marks all max positions)
                    mask[i, c, hs:he, ws:we] |= (region == max_val)

    cache = (X.shape, size, stride, mask)
    return P, cache


def flatten_forward(X):
    # X: (m, F, H, W) -> (m, F*H*W)
    return X.reshape(X.shape[0], -1)

def dense_forward(X, W, b):
    """
    X: (m, in_dim)
    W: (out_dim, in_dim)
    W.T: (in_dim, out_dim)
    b: (out_dim,)
    returns Z: (m, out_dim)
    """
    return X.dot(W.T) + b

def cnn_forward(X, Wc, bc, Wd, bd):
    # X: (m, 1, 28, 28)
    Zc = conv_forward(X, Wc, bc, stride=1, pad=1)   # (m, F, 28, 28)
    Ac = ReLU(Zc)                                   # (m, F, 28, 28)
    P, pool_cache = maxpool_forward(Ac, size=2, stride=2)      # (m, F, 14, 14)
    Xf = flatten_forward(P)                         # (m, F*14*14)
    Z  = dense_forward(Xf, Wd, bd)                  # (m, 10)
    A2 = softmax(Z)                                 # (m, 10)
    cache = (X, Zc, Ac, P, Xf, Z, pool_cache)       # saving for backprop later
    return A2, cache

# End of CNN forward prop

def one_hot(Y, num_classes=10):
    Y = Y.astype(int)   
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

# Start of CNN backward prop

def backward_head_and_pool(A2, Y, cache, Wd, lambd=0.0):
    """
    Backprop through: Softmax+CE + Dense + Unflatten + MaxPool
    Returns:
      dAc  : (m, F, 28, 28) gradient wrt Ac (output of ReLU after conv)
      dWd  : (10, flat_dim)
      dbd  : (10,)
    """
    X, Zc, Ac, P, Xf, Z, pool_cache = cache
    m = Y.shape[0]

    # (1) softmax + cross-entropy gradient
    dZ = A2.copy()                  # (m,10)
    dZ[np.arange(m), Y] -= 1
    dZ /= m

    # (2) dense backward
    dWd = dZ.T @ Xf + (lambd/m) * Wd   # (10, flat_dim)
    dbd = np.sum(dZ, axis=0)           # (10,)
    dXf = dZ @ Wd                      # (m, flat_dim)

    # (3) unflatten
    dP = dXf.reshape(P.shape)          # (m, F, 14, 14)

    # (4) pool backward
    dAc = maxpool_backward(dP, pool_cache)  # (m, F, 28, 28)

    return dAc, dWd, dbd

def maxpool_backward(dP, cache):
    (X_shape, size, stride, mask) = cache
    m, C, H, W = X_shape
    _, _, H_out, W_out = dP.shape

    dX = np.zeros((m, C, H, W), dtype=dP.dtype)

    for i in range(m):
        for c in range(C):
            for h in range(H_out):
                hs = h * stride
                he = hs + size
                for w in range(W_out):
                    ws = w * stride
                    we = ws + size

                    # distribute gradient to max positions
                    region_mask = mask[i, c, hs:he, ws:we]
                    dX[i, c, hs:he, ws:we] += dP[i, c, h, w] * region_mask

    return dX

def relu_backward(dA, Z):
    # dA and Z have same shape
    return dA * (Z > 0)

def conv_backward(dZ, X, W, pad=1):
    """
    dZ: (m, F, H_out, W_out)  gradient wrt conv output
    X : (m, C_in, H, W_in)    original conv input
    W : (F, C_in, K, K)       conv filters
    returns: dX, dW, db
    """
    m, C_in, H, W_in = X.shape
    F, _, K, _ = W.shape
    _, _, H_out, W_out = dZ.shape

    X_pad = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode="constant")
    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)
    db = np.zeros((F,), dtype=dZ.dtype)

    # db: sum over batch and spatial dims
    db = np.sum(dZ, axis=(0,2,3))  # (F,)

    for i in range(m):
        for f in range(F):
            for h in range(H_out):
                hs = h  # stride=1
                he = hs + K
                for w in range(W_out):
                    ws = w  # stride=1
                    we = ws + K

                    region = X_pad[i, :, hs:he, ws:we]   # (C_in,K,K)

                    # dW accumulates input patch scaled by dZ
                    dW[f] += region * dZ[i, f, h, w]

                    # dX accumulates filter scaled by dZ
                    dX_pad[i, :, hs:he, ws:we] += W[f] * dZ[i, f, h, w]

    # remove padding
    dX = dX_pad[:, :, pad:pad+H, pad:pad+W_in]
    return dX, dW, db

def backward_full(A2, Y, cache, Wd, Wc, lambd=0.0):
    X, Zc, Ac, P, Xf, Z, pool_cache = cache
    m = Y.shape[0]

    # softmax+CE
    dZ = A2.copy()
    dZ[np.arange(m), Y] -= 1
    dZ /= m

    # dense backward
    dWd = dZ.T @ Xf + (lambd/m) * Wd
    dbd = np.sum(dZ, axis=0)
    dXf = dZ @ Wd

    # unflatten
    dP = dXf.reshape(P.shape)

    # pool backward -> dAc
    dAc = maxpool_backward(dP, pool_cache)

    # relu backward -> dZc
    dZc = relu_backward(dAc, Zc)

    # conv backward -> dX, dWc, dbc
    dX, dWc, dbc = conv_backward(dZc, X, Wc, pad=1)

    return dWc, dbc, dWd, dbd

# End of CNN backward prop

 # Staart of training loop
def update_params_cnn(Wc, bc, Wd, bd, dWc, dbc, dWd, dbd, alpha):
    Wc -= alpha * dWc
    bc -= alpha * dbc
    Wd -= alpha * dWd
    bd -= alpha * dbd
    return Wc, bc, Wd, bd


def get_predictions(A2):
    return np.argmax(A2, axis = 1)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def compute_loss(A2, Y, eps=1e-12):
    m = Y.shape[0]
    return -np.mean(np.log(A2[np.arange(m), Y] + eps))

import time

def train_cnn_batches(X, Y, alpha=0.01, epochs=2, batch_size=256,lambd=0.0, num_filters=8,kernel_size=3,seed=42):
    rng = np.random.default_rng(seed)
    Wc, bc, Wd, bd = init_params_cnn(num_filters=num_filters, kernel_size=kernel_size)

    n = X.shape[0]
    steps = 0

    for ep in range(epochs):
        perm = rng.permutation(n)
        Xs = X[perm]
        Ys = Y[perm]

        for start in range(0, n, batch_size):
            t0 = time.time()

            end = min(start + batch_size, n)
            Xb = Xs[start:end]
            Yb = Ys[start:end]

            # forward + backward + update on the batch
            A2, cache = cnn_forward(Xb, Wc, bc, Wd, bd)
            dWc, dbc, dWd, dbd = backward_full(A2, Yb, cache, Wd, Wc, lambd=lambd)
            Wc, bc, Wd, bd = update_params_cnn(Wc, bc, Wd, bd, dWc, dbc, dWd, dbd, alpha)

            # prints: first 2 steps + every 20 steps
            if steps < 2 or steps % 20 == 0:
                loss = compute_loss(A2, Yb)  # batch loss
                acc = np.mean(np.argmax(A2, axis=1) == Yb)
                print(f"Ep {ep} Step {steps}: batch_loss={loss:.4f}, batch_acc={acc:.4f}, time={time.time()-t0:.2f}s")

            steps += 1

        # end-of-epoch eval on ALL X (expensive) OR on a small fixed sample (recommended)
        eval_idx = rng.choice(n, size=min(256, n), replace=False)
        A2e, _ = cnn_forward(X[eval_idx], Wc, bc, Wd, bd)
        loss_e = compute_loss(A2e, Y[eval_idx])
        acc_e = np.mean(np.argmax(A2e, axis=1) == Y[eval_idx])
        print(f"== Epoch {ep} done: eval_loss={loss_e:.4f}, eval_acc={acc_e:.4f} ==")

    return Wc, bc, Wd, bd


rng = np.random.default_rng(42)
idx = rng.choice(X_train_cnn.shape[0], 1024, replace=False)
X_dbg = X_train_cnn[idx]
Y_dbg = Y_train[idx].astype(int)

Wc, bc, Wd, bd = train_cnn_batches(
    X_train_cnn, Y_train.astype(int),
    alpha=0.01,
    epochs=3,          # start small because it's slow
    batch_size=256,
    lambd=0.0,
    num_filters=8
)


def evaluate_model_cnn(Wc, bc, Wd, bd, X_val_cnn, Y_val):
    A2, _ = cnn_forward(X_val_cnn, Wc, bc, Wd, bd)
    predictions = np.argmax(A2, axis=1)

    print("Predictions:", predictions[:10])
    print("Y_val      :", Y_val[:10])
    print("Shape of predictions:", predictions.shape)
    print("Shape of Y_val      :", Y_val.shape)

    acc = np.mean(predictions == Y_val.astype(int))
    return acc

dev_acc_200 = evaluate_model_cnn(Wc, bc, Wd, bd, X_dev_cnn[:200], Y_dev[:200])
test_acc_200 = evaluate_model_cnn(Wc, bc, Wd, bd, X_test_cnn[:200], Y_test[:200])
print("Dev Accuracy:", dev_acc_200)
print("Test Accuracy:", test_acc_200)

def make_predictions_cnn(X_cnn, Wc, bc, Wd, bd):
    A2, _ = cnn_forward(X_cnn, Wc, bc, Wd, bd)
    return np.argmax(A2, axis=1)

def test_prediction_cnn(index, Wc, bc, Wd, bd):
    current_image = X_train_cnn[index:index+1]  # keep batch dim -> (1,1,28,28)
    prediction = make_predictions_cnn(current_image, Wc, bc, Wd, bd)[0]
    label = int(Y_train[index])

    print("Prediction:", prediction)
    print("Label     :", label)

    img = current_image[0, 0]  # (28,28)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


for i in range(5, 1000, 100):
    test_prediction_cnn(i, Wc, bc, Wd, bd)


#This is the full CNN for MNIST, full evaluation will take 3 hours, switching to PyTorch in next file for faster training and better GPU support