import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1) Load MNIST CSV + split
# -----------------------------
data = pd.read_csv(r"C:\Users\yoavn\Downloads\archive\train.csv").to_numpy()
np.random.shuffle(data)

data_dev = data[0:1000]
Y_dev = data_dev[:, 0].astype(np.int64)
X_dev = data_dev[:, 1:].astype(np.float32) / 255.0

data_test = data[1000:2000]
Y_test = data_test[:, 0].astype(np.int64)
X_test = data_test[:, 1:].astype(np.float32) / 255.0

data_train = data[2000:]
Y_train = data_train[:, 0].astype(np.int64)
X_train = data_train[:, 1:].astype(np.float32) / 255.0

X_train_cnn = X_train.reshape(-1, 1, 28, 28)
X_dev_cnn   = X_dev.reshape(-1, 1, 28, 28)
X_test_cnn  = X_test.reshape(-1, 1, 28, 28)

# -----------------------------
# 2) Dataloaders
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_ds = TensorDataset(torch.from_numpy(X_train_cnn), torch.from_numpy(Y_train))
dev_ds   = TensorDataset(torch.from_numpy(X_dev_cnn),   torch.from_numpy(Y_dev))
test_ds  = TensorDataset(torch.from_numpy(X_test_cnn),  torch.from_numpy(Y_test))

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

# -----------------------------
# 3) Model: conv -> relu -> conv -> relu -> pool -> flatten -> dense
# -----------------------------
class CNN2Conv(nn.Module):
    def __init__(self, num_filters=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)          # (N, F, 28, 28)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1) # (N, F, 28, 28)
        self.pool  = nn.MaxPool2d(2, 2)                                          # (N, F, 14, 14)
        self.fc    = nn.Linear(num_filters * 14 * 14, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

model = CNN2Conv(num_filters=32).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # strong default for MNIST

# -----------------------------
# 4) Train + eval
# -----------------------------
@torch.no_grad()
def evaluate(loader, name="Eval"):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    print(f"{name}: loss={running_loss/total:.4f}, acc={correct/total:.4f}")
    return correct / total

@torch.no_grad()
def evaluate_return(loader):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    return running_loss/total, correct/total

# BEFORE training starts
train_loss0, train_acc0 = evaluate_return(train_loader)  # or a subset loader if you want faster
dev_loss0, dev_acc0     = evaluate_return(dev_loader)

history = {"train_loss":[train_loss0], "train_acc":[train_acc0],
           "dev_loss":[dev_loss0],     "dev_acc":[dev_acc0]}

def train(epochs=10):
    history = {
        "train_loss": [], "train_acc": [],
        "dev_loss": [], "dev_acc": [],
        "batch_loss": [], "batch_acc": []   # <-- FIX 3
    }

    global_step = 0

    for ep in range(epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # --- per-batch tracking (FIX 3) ---
            history["batch_loss"].append(loss.item())
            with torch.no_grad():
                preds_b = logits.argmax(dim=1)
                acc_b = (preds_b == yb).float().mean().item()
            history["batch_acc"].append(acc_b)
            global_step += 1
            # ----------------------------------

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        dev_loss, dev_acc = evaluate_return(dev_loader)

        print(
            f"Epoch {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)

    return history

def plot_batch_learning(history):
    steps = np.arange(len(history["batch_loss"]))

    plt.figure()
    plt.plot(steps, history["batch_loss"])
    plt.xlabel("Step (batch)")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Batch Loss)")
    plt.show()

    plt.figure()
    plt.plot(steps, history["batch_acc"])
    plt.xlabel("Step (batch)")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Batch Accuracy)")
    plt.ylim(0, 1)
    plt.show()


history = train(epochs=10)
plot_batch_learning(history)
evaluate(test_loader, name="Test")

# -----------------------------
# 5) Single-image prediction + plot
# -----------------------------
@torch.no_grad()
def test_prediction(index):
    model.eval()
    x = torch.from_numpy(X_train_cnn[index:index+1]).to(device)
    y = int(Y_train[index])

    logits = model(x)
    pred = int(logits.argmax(dim=1).item())

    print("Prediction:", pred)
    print("Label     :", y)

    plt.imshow(X_train_cnn[index, 0], cmap="gray")
    plt.axis("off")
    plt.show()

for i in range(5, 1000, 100):
    test_prediction(i)


def plot_learning_curves(history):
    epochs = np.arange(0, len(history["train_acc"])) 

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["dev_acc"], label="Dev Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Accuracy)")
    lo = min(min(history["train_acc"]), min(history["dev_acc"])) - 0.01
    plt.ylim(max(0, lo), 1.0)
    plt.legend()
    plt.show()

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["dev_loss"], label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()
    plt.show()

plot_learning_curves(history)
