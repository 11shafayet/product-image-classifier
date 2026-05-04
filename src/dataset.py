import os                            # os helps us join folder path + image filename
from PIL import Image                # PIL.Image opens image files like: 15970.jpg

import torch                         # importing pytorch
from torch.utils.data import Dataset # We inherit from it to create our own dataset
from torchvision import transforms   # Used to preprocess images before sending them to the model

class ProductDataset(Dataset):
    def __init__(self, dataframe, image_dir, train=True):
        
        self.dataframe = dataframe
        self.image_dir = image_dir

        # Training transforms (WITH augmentation)
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomRotation(10),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),

                transforms.ToTensor()
            ])

        # Validation transforms (NO augmentation)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.dataframe)              # This tells pyTorch how many samples are in the dataset
    
    def __getitem__(self, idx):               # This tells PyTorch how to get ONE item from the dataset.

        image_name = self.dataframe.iloc[idx]["image_path"]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        label = self.dataframe.iloc[idx]["label"]

        return image, label