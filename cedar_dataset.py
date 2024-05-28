import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CedarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assuming two directories: 'genuine' and 'forgery'
        genuine_dir = os.path.join(root_dir, "genuine")
        forgery_dir = os.path.join(root_dir, "forgery")

        for filename in os.listdir(genuine_dir):
            self.image_paths.append(os.path.join(genuine_dir, filename))
            self.labels.append(1)

        for filename in os.listdir(forgery_dir):
            self.image_paths.append(os.path.join(forgery_dir, filename))
            self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
