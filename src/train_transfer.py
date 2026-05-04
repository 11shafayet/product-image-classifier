import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models

from dataset import ProductDataset


CSV_PATH = "data/raw/fashion-product-images/styles.csv"
IMAGE_DIR = "data/raw/fashion-product-images/images"

# ---------------- LOAD DATA ----------------

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

df = df.dropna(subset=["masterCategory"])

df["image_path"] = df["id"].astype(str) + ".jpg"

existing_images = []

for img_name in df["image_path"]:

    img_path = os.path.join(IMAGE_DIR, img_name)

    if os.path.exists(img_path):
        existing_images.append(True)
    else:
        existing_images.append(False)

df = df[existing_images]

main_categories = [
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care"
]

df = df[df["masterCategory"].isin(main_categories)]

# Balanced dataset
# sampled_dfs = []

# for category in main_categories:

#     category_df = df[df["masterCategory"] == category]

#     sample_size = min(len(category_df), 2000)

#     sampled_dfs.append(
#         category_df.sample(sample_size, random_state=42)
#     )

# df = pd.concat(sampled_dfs).reset_index(drop=True)

# print(df["masterCategory"].value_counts())

# Use all available images from selected categories
df = df.reset_index(drop=True)

print(df["masterCategory"].value_counts())
print("Total images used:", len(df))

# ---------------- LABELS ----------------

class_names = sorted(df["masterCategory"].unique())

class_to_idx = {}

for index, class_name in enumerate(class_names):
    class_to_idx[class_name] = index

df["label"] = df["masterCategory"].map(class_to_idx)

# ---------------- SPLIT ----------------

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["masterCategory"]
)

# ---------------- DATASET ----------------

train_dataset = ProductDataset(
    train_df,
    IMAGE_DIR,
    train=True
)

val_dataset = ProductDataset(
    val_df,
    IMAGE_DIR,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ---------------- DEVICE ----------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ---------------- MODEL ----------------

model = models.resnet18(weights="DEFAULT")

# Freeze all pretrained layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last ResNet block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model = model.to(device)

# ---------------- LOSS ----------------

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=0.0001
)

# ---------------- TRAIN ----------------

num_epochs = 3

best_accuracy = 0.0

for epoch in range(num_epochs):

    # TRAIN
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

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "final_resnet18_product_classifier.pth"
        )

        print(f"\nFinal best model saved! Accuracy: {accuracy:.2f}%")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

# ---------------- REPORT ----------------

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)

print(f"\nBest Accuracy: {best_accuracy:.2f}%")