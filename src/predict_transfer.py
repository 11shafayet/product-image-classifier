import argparse
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


def predict(image_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = class_names[predicted.item()]

    confidence_score = confidence.item() * 100

    print(f"\nImage: {image_path}")

    print(f"Predicted class: {predicted_class}")

    print(f"Confidence: {confidence_score:.2f}%")

    print("\nAll probabilities:\n")

    for class_name, prob in zip(class_names, probabilities[0]):

        print(
            f"{class_name}: {prob.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    predict(args.image)