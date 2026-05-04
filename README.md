# Product Image Classifier

A deep learning image classification project using the Kaggle Fashion Product Images dataset.

Dataset link: [text](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data)

The goal of this project was to classify product images into 4 major categories:

- Accessories
- Apparel
- Footwear
- Personal Care

The project started with a custom CNN from scratch and later improved using transfer learning with ResNet18.

---

# 1. Project Setup

Project folder:

```text
product-image-classifier/
│
├── data/
│   └── raw/
│       └── fashion-product-images/
│           ├── images/
│           └── styles.csv
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── batch_predict.py
│   ├── train_transfer.py
│   ├── predict_transfer.py
│   └── batch_predict_transfer.py
│
├── test_images/
├── final_resnet18_product_classifier.pth
└── README.md
```

---

# 2. Dataset

Dataset used:

```text
Fashion Product Images Small
```

Main files:

```text
styles.csv
images/
```

The CSV contains product metadata. The image folder contains product images named by product ID.

Example:

```text
id = 15970
image = 15970.jpg
```

---

# 3. Initial Data Loading

I first loaded the CSV file:

```python
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
```

Then removed rows with missing labels:

```python
df = df.dropna(subset=["masterCategory"])
```

Then created image filenames:

```python
df["image_path"] = df["id"].astype(str) + ".jpg"
```

Then kept only rows where the image file actually existed.

---

# 4. Selected Classes

I did not train on every category at first.

I selected 4 main product categories:

```python
main_categories = [
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care"
]
```

This made the project easier to train, evaluate, and understand.

---

# 5. First Training Experiment: 4000 Images

At first, I used a small balanced dataset:

```text
Accessories      1000
Apparel          1000
Footwear         1000
Personal Care    1000
```

Total:

```text
4000 images
```

This was done to make training faster.

Result:

```text
Validation Accuracy: around 89%–93%
```

This proved the full training pipeline was working.

---

# 6. Evaluation Metrics Added

Then added:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Using:

```python
from sklearn.metrics import classification_report, confusion_matrix
```

This helped to understand class-wise performance.

Example evaluation:

```text
precision
recall
f1-score
support
```

Important observation:

```text
Apparel and Accessories were often confused.
```

---

# 7. Single Image Prediction

Then created:

```text
src/predict.py
```

This allowed prediction on one image:

```bash
py src/predict.py --image test_image.jpg
```

Example output:

```text
Predicted class: Apparel
Confidence: 89.45%

Accessories: 10.40%
Apparel: 89.45%
Footwear: 0.06%
Personal Care: 0.09%
```

---

# 8. Batch Prediction

Then created:

```text
src/batch_predict.py
```

This allowed prediction on multiple images from a folder:

```bash
py src/batch_predict.py --folder test_images
```

Example output:

```text
Image                          Prediction      Confidence
1535.jpg                       Apparel         67.46%
1549.jpg                       Accessories     50.47%
1575.jpg                       Apparel         99.61%
```

Then I added low-confidence warning:

```python
if confidence < 70:
    status = "LOW CONFIDENCE"
```

---

# 9. Increased Dataset Size: 8000 Images

Then tried to increase the sample size:

```text
2000 images per class
```

Total:

```text
8000 images
```

Result:

```text
Validation Accuracy: 95.50%
```

This showed an important ML lesson:

```text
More real data improved performance more than model changes.
```

---

# 10. Data Augmentation Experiment

THen I added training augmentation in `dataset.py`:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(10)
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2
)
```

Validation images were kept clean:

```python
transforms.Resize((128, 128))
transforms.ToTensor()
```

Result:

```text
Validation Accuracy: 94.88%
```

Accuracy slightly dropped, but the model became more robust.

Important lesson:

```text
Augmentation can improve generalization, even if validation accuracy does not always increase immediately.
```

---

# 11. BatchNorm + Dropout Experiment

After that, I upgraded the custom CNN with:

- Batch Normalization
- Dropout
- More convolution blocks

First larger model:

```text
Conv 32 → 64 → 128 → 256
Dropout 0.4
```

But it became unstable.

Example result:

```text
Epoch 1 Accuracy: 83.19%
Epoch 2 Accuracy: 93.69%
Epoch 3 Accuracy: 57.94%
```

Problem:

```text
Training collapsed and the model predicted mostly Apparel.
```

Then we tried a safer model:

```text
Conv 32 → 64 → 128
Dropout 0.25
```

Result:

```text
Validation Accuracy: 93.31%
```

Conclusion:

```text
The simpler CNN was better for this dataset.
Bigger model does not always mean better model.
```

---

# 12. Best Model Checkpointing

I added best model saving.

Instead of saving the final epoch, I saved only the best validation model:

```python
if accuracy > best_accuracy:
    best_accuracy = accuracy

    torch.save(
        model.state_dict(),
        "best_product_classifier.pth"
    )
```

This is called:

```text
Model Checkpointing
```

It prevents a worse final epoch from replacing the best model.

---

# 13. Transfer Learning with ResNet18

After the custom CNN, I created:

```text
src/train_transfer.py
```

I used ResNet18 from torchvision:

```python
model = models.resnet18(weights="DEFAULT")
```

This loaded pretrained ImageNet weights.

---

# 14. First Transfer Learning Attempt: Frozen ResNet18

First, froze all pretrained layers:

```python
for param in model.parameters():
    param.requires_grad = False
```

Then replaced the final classification layer:

```python
model.fc = nn.Linear(num_features, len(class_names))
```

Only the final layer was trained.

Result:

```text
Validation Accuracy: 90.44%
```

Conclusion:

```text
Frozen ResNet worked, but it did not adapt deeply to the product image dataset.
```

---

# 15. Fine-Tuning ResNet18

Then unfroze the final ResNet block:

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

And trained both:

```text
layer4 + fc
```

Optimizer:

```python
optimizer = optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=0.0001
)
```

Result on 8000 balanced images:

```text
Best Accuracy: 99.06%
```

This was a major improvement.

---

# 16. Final Full Dataset Training

Finally, trained fine-tuned ResNet18 on all available selected-category images.

Class distribution:

```text
Apparel          21392
Accessories      11274
Footwear          9219
Personal Care     2403
```

Total:

```text
44,288 images
```

Validation set:

```text
8,858 images
```

Final result:

```text
Best Accuracy: 99.30%
Macro F1-score: 0.99
Weighted F1-score: 0.99
```

Final classification report:

```text
               precision    recall  f1-score   support

Accessories       0.99      0.99      0.99      2255
Apparel           0.99      1.00      1.00      4278
Footwear          1.00      0.99      1.00      1844
Personal Care     0.99      0.98      0.99       481

accuracy                              0.99      8858
macro avg          0.99      0.99      0.99      8858
weighted avg       0.99      0.99      0.99      8858
```

The final model was saved as:

```text
final_resnet18_product_classifier.pth
```

---

# 17. Final Transfer Prediction Script

Then created:

```text
src/predict_transfer.py
```

Run:

```bash
py src/predict_transfer.py --image test_image.jpg
```

Example output:

```text
Image: test_image.jpg
Predicted class: Apparel
Confidence: 92.72%

Accessories: 4.02%
Apparel: 92.72%
Footwear: 0.79%
Personal Care: 2.47%
```

---

# 18. Final Batch Transfer Prediction

Then created:

```text
src/batch_predict_transfer.py
```

Run:

```bash
py src/batch_predict_transfer.py --folder test_images
```

Example output:

```text
Image                         Prediction        Confidence     Status
1535.jpg                      Accessories       93.75%        OK
1549.jpg                      Footwear          99.81%        OK
1575.jpg                      Apparel           99.76%        OK
1599.jpg                      Accessories       93.05%        OK
1609.jpg                      Apparel           99.74%        OK
```

---

# 19. Removed / Replaced Experiments

During development, some approaches were tested and then replaced.

## Removed: 4000-image limited training

Reason:

```text
Good for testing, but not final performance.
```

## Removed: Large custom CNN with 256 channels

Reason:

```text
Training became unstable and collapsed.
```

## Removed: Frozen-only ResNet18 as final model

Reason:

```text
Accuracy was only 90.44%.
Fine-tuning layer4 performed much better.
```

## Removed: final-epoch model saving

Reason:

```text
Best validation epoch is more reliable than final epoch.
```

Replaced with:

```text
Best model checkpointing
```

---

# 20. Final Model Choice

Final selected model:

```text
Fine-tuned ResNet18
```

Final model file:

```text
final_resnet18_product_classifier.pth
```

Why this model was selected:

- Highest accuracy
- Strong class-wise F1-score
- Stable training
- Excellent batch prediction confidence
- Industry-standard transfer learning approach

---

# 21. Key Learnings

This project covered:

- Image classification
- CNN from scratch
- Dataset cleaning
- Label encoding
- Train-validation split
- DataLoader
- Training loop
- Validation loop
- Classification report
- Confusion matrix
- Softmax probabilities
- Single-image prediction
- Batch prediction
- Confidence thresholding
- Data augmentation
- Batch Normalization
- Dropout
- Model checkpointing
- Transfer learning
- ResNet18 fine-tuning

---

# 23. Final Result

```text
Final Accuracy: 99.30%
Final Macro F1-score: 0.99
Final Weighted F1-score: 0.99
```
