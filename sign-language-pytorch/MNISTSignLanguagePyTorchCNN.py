# sign_pytorch.py
# PyTorch training for Sign Language MNIST 
# Uses BetterSignCNN + proper label remapping + torchvision-style augmentations
# Adds: dataset preview, label histogram, training curves, confusion matrix, and PNG prediction visualizations

import string
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torch.optim import AdamW

import matplotlib.pyplot as plt

TRAIN_CSV = r"C:\Users\yoavn\Downloads\archive (1)\sign_mnist_train.csv"
TEST_CSV  = r"C:\Users\yoavn\Downloads\archive (1)\sign_mnist_test.csv"

IMG1 = r"C:\Users\yoavn\Downloads\archive (1)\american_sign_language.PNG"
IMG2 = r"C:\Users\yoavn\Downloads\archive (1)\amer_sign2.png"
IMG3 = r"C:\Users\yoavn\Downloads\archive (1)\amer_sign3.png"

def remap_label_kaggle_to_0_23(y: int) -> int:
    return y - 1 if y > 9 else y

# 24 letters: A..Y without J (Z excluded in this dataset due to motion)
LETTERS = [c for c in string.ascii_uppercase if c not in ["J", "Z"]][:24]

class SignMNISTCsv(Dataset):
    def __init__(self, csv_path: str, train: bool):
        df = pd.read_csv(csv_path)

        y_raw = df["label"].astype(int).to_numpy()
        X = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)  # 0..255 pixels
        X = X.reshape(-1, 28, 28)  # grayscale

        # remap labels to 0..23
        y = np.array([remap_label_kaggle_to_0_23(int(v)) for v in y_raw], dtype=np.int64)

        self.X = X
        self.y = y
        self.train = train

        if train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=0
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),  # float32 [0,1], shape [1,28,28]
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]                 # (28,28) uint8
        label = int(self.y[idx])          # 0..23
        x = self.transform(img)           # FloatTensor [1,28,28]
        y = torch.tensor(label, dtype=torch.long)
        return x, y

class BetterSignCNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 28 -> 14
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 14 -> 7
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, ceil_mode=True),  # 7 -> 4 safely
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> 128 x 1 x 1
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)  # logits (NO softmax)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).float().sum().item()
        n += bs

    return total_loss / n, total_correct / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).float().sum().item()
        n += bs

    return total_loss / n, total_correct / n

def show_dataset_preview(train_ds: SignMNISTCsv, indices=(0, 1, 2, 4)):
    fig, axes = plt.subplots(2, 2)
    fig.suptitle("Preview of dataset (train)")

    for ax, idx in zip(axes.ravel(), indices):
        img = train_ds.X[idx]  # raw uint8 28x28
        label = int(train_ds.y[idx])
        ax.imshow(img, cmap="gray")
        ax.set_title(f"label: {label}  letter: {LETTERS[label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_label_frequency(train_ds: SignMNISTCsv):
    counts = np.bincount(train_ds.y, minlength=24)

    plt.figure()
    plt.bar(np.arange(24), counts)
    plt.title("Frequency of each label (train)")
    plt.xticks(np.arange(24), LETTERS, rotation=0)
    plt.xlabel("Class (letter)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_training_curves(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["test_loss"], label="test loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, np.array(history["train_acc"]) * 100, label="train acc (%)")
    plt.plot(epochs, np.array(history["test_acc"]) * 100, label="test acc (%)")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def compute_confusion_matrix(model, loader, device, num_classes=24):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.numpy()
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        for t, p in zip(y, preds):
            cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test)")
    plt.colorbar()
    plt.xticks(np.arange(24), LETTERS, rotation=90)
    plt.yticks(np.arange(24), LETTERS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def predict_image(model, img_path, device):
    model.eval()

    tfm = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((28, 28)),
        T.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # [1,1,28,28]
    logits = model(x)
    pred = logits.argmax(dim=1).item()

    return pred, LETTERS[pred], img

def show_png_predictions(model, device, paths):
    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 4))
    if len(paths) == 1:
        axes = [axes]

    for ax, p in zip(axes, paths):
        try:
            idx, letter, img = predict_image(model, p, device)
            ax.imshow(img)
            ax.set_title(f"Pred: {letter} (class {idx})")
            ax.axis("off")
        except Exception as e:
            ax.set_title(f"Error\n{e}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = SignMNISTCsv(TRAIN_CSV, train=True)
    test_ds  = SignMNISTCsv(TEST_CSV,  train=False)

    # Visuals like the Kaggle notebook
    show_dataset_preview(train_ds)
    plot_label_frequency(train_ds)

    # Loaders (num_workers=0 safest on Windows)
    train_loader = DataLoader(train_ds, batch_size=200, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=400, shuffle=False, num_workers=0)

    model = BetterSignCNN(num_classes=24).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 15

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_one_epoch(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
            f"test loss {te_loss:.4f} acc {te_acc*100:.2f}%"
        )

    print(f"\nFINAL TEST ACCURACY = {history['test_acc'][-1]*100:.2f}%")

    # Training curves
    plot_training_curves(history)

    # Confusion matrix
    cm = compute_confusion_matrix(model, test_loader, device, num_classes=24)
    plot_confusion_matrix(cm)

    # Show PNG predictions (your sample images)
    show_png_predictions(model, device, [IMG1, IMG2, IMG3])


if __name__ == "__main__":
    main()
