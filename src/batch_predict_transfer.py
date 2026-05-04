import argparse
import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


MODEL_PATH = "final_resnet18_product_classifier.pth"

class_names = [
    "Accessories",
    "Apparel",
    "Footwear",
    "Personal Care"
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def load_model(device):

    model = models.resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(class_names)
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model = model.to(device)

    model.eval()

    return model


def predict_image(image_path, model, device):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    predicted_class = class_names[predicted_index]

    confidence = (
        probabilities[0][predicted_index].item() * 100
    )

    return predicted_class, confidence


def main(folder_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    valid_extensions = [".jpg", ".jpeg", ".png"]

    image_files = [
        file for file in os.listdir(folder_path)
        if os.path.splitext(file.lower())[1]
        in valid_extensions
    ]

    print("\nBatch Prediction Results")
    print("-" * 75)

    print(
        f"{'Image':30}"
        f"{'Prediction':18}"
        f"{'Confidence':15}"
        f"{'Status'}"
    )

    print("-" * 75)

    for image_file in image_files:

        image_path = os.path.join(
            folder_path,
            image_file
        )

        predicted_class, confidence = predict_image(
            image_path,
            model,
            device
        )

        status = "OK"

        if confidence < 70:
            status = "LOW CONFIDENCE"

        print(
            f"{image_file:30}"
            f"{predicted_class:18}"
            f"{confidence:.2f}%{'':8}"
            f"{status}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing images"
    )

    args = parser.parse_args()

    main(args.folder)