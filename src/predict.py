import argparse
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


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProductCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_index]
    confidence = probabilities[0][predicted_index].item() * 100

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll probabilities:")

    for class_name, probability in zip(class_names, probabilities[0]):
        print(f"{class_name}: {probability.item() * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to product image"
    )

    args = parser.parse_args()

    predict(args.image)