import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import ProductDataset
from model import ProductCNN


CSV_PATH = "data/raw/fashion-product-images/styles.csv"
IMAGE_DIR = "data/raw/fashion-product-images/images"

# Load CSV
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Remove rows with missing labels
df = df.dropna(subset=["masterCategory"])

# Create image path column
df["image_path"] = df["id"].astype(str) + ".jpg"

# Keep only rows where image exists
existing_images = []

for img_name in df["image_path"]:
    img_path = os.path.join(IMAGE_DIR, img_name)

    if os.path.exists(img_path):
        existing_images.append(True)
    else:
        existing_images.append(False)

df = df[existing_images]

# Remove very small classes
category_counts = df["masterCategory"].value_counts()

valid_categories = category_counts[category_counts >= 2].index

df = df[df["masterCategory"].isin(valid_categories)]

main_categories = [
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care"
]

df = df[df["masterCategory"].isin(main_categories)]

# Use small balanced dataset first for faster CPU training
sampled_dfs = []

for category in main_categories:
    category_df = df[df["masterCategory"] == category]
    sample_size = min(len(category_df), 2000)
    sampled_dfs.append(category_df.sample(sample_size, random_state=42))

df = pd.concat(sampled_dfs).reset_index(drop=True)

# Create label mapping
class_names = sorted(df["masterCategory"].unique())

class_to_idx = {}

for index, class_name in enumerate(class_names):
    class_to_idx[class_name] = index

df["label"] = df["masterCategory"].map(class_to_idx)

# Train-validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["masterCategory"]
)

train_dataset = ProductDataset(train_df, IMAGE_DIR, train=True)
val_dataset = ProductDataset(val_df, IMAGE_DIR, train=False)

# Data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
num_classes = len(class_names)

model = ProductCNN(num_classes=num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0003
)

# Training settings
num_epochs = 3
best_accuracy = 0.0

# Training loop
for epoch in range(num_epochs):

    # ---------------- TRAINING ----------------
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Save best model only
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "best_product_classifier.pth"
        )

        print(f"\nBest model saved! Accuracy: {accuracy:.2f}%")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

print(f"\nBest validation accuracy: {best_accuracy:.2f}%")

# ---------------- FINAL EVALUATION ----------------

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(all_labels, all_predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()