import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image, UnidentifiedImageError
import os

class CedarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assuming two directories: 'full_org' and 'full_forg'
        genuine_dir = os.path.join(root_dir, "full_org")
        forgery_dir = os.path.join(root_dir, "full_forg")

        for filename in os.listdir(genuine_dir):
            file_path = os.path.join(genuine_dir, filename)
            if self.is_image_file(file_path):
                self.image_paths.append(file_path)
                self.labels.append(1)

        for filename in os.listdir(forgery_dir):
            file_path = os.path.join(forgery_dir, filename)
            if self.is_image_file(file_path):
                self.image_paths.append(file_path)
                self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("L")  # Convert to grayscale
        except UnidentifiedImageError:
            # In case an image cannot be opened, you can handle it here
            print(f"Cannot identify image file {img_path}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def is_image_file(self, file_path):
        try:
            Image.open(file_path).verify()
            return True
        except (IOError, UnidentifiedImageError):
            return False