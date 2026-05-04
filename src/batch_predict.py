import argparse
import os
import torch
from PIL import Image
from torchvision import transforms

from model import ProductCNN


MODEL_PATH = "best_product_classifier.pth"

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


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProductCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[0][predicted_index].item() * 100

    return predicted_class, confidence


def main(folder_path):
    model, device = load_model()

    valid_extensions = [".jpg", ".jpeg", ".png"]

    image_files = [
        file for file in os.listdir(folder_path)
        if os.path.splitext(file.lower())[1] in valid_extensions
    ]

    if len(image_files) == 0:
        print("No image files found.")
        return

    print("\nBatch Prediction Results")
    print("-" * 60)
    print(f"{'Image':30} {'Prediction':15} {'Confidence'}")
    print("-" * 60)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        predicted_class, confidence = predict_image(
            image_path,
            model,
            device
        )

        status = "OK"

        if confidence < 70:
            status = "LOW CONFIDENCE"

        print(f"{image_file:30} {predicted_class:15} {confidence:.2f}%   {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing product images"
    )

    args = parser.parse_args()

    main(args.folder)