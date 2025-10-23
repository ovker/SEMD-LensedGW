# dataset.py
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LensedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # ensure 3-channel RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(data_dirs, split_ratios=(0.5, 0.125, 0.375), random_seed=42):
    from sklearn.model_selection import train_test_split

    image_paths = []
    labels = []
    # Gather file paths and labels
    for label_name, folder in data_dirs.items():
        label = 1 if label_name.lower().startswith('lensed') else 0
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder, fname))
                labels.append(label)

    # Split into train+val and test first
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=split_ratios[2], random_state=random_seed, stratify=labels)
    # Split train and val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]),
        random_state=random_seed, stratify=train_val_labels)

    splits = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }
    return splits

def load_data_from_dirs(data_dirs, split_ratios=(0.7, 0.15, 0.15), random_seed=42):
    return load_data(data_dirs, split_ratios, random_seed)

# Define transforms for train, val, test
def get_transforms():
    """Return torchvision transforms for preprocessing."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])  # ImageNet stds
    ])
    return transform



def crop_region(img):
    return img.crop((600, 200, 2225, 1356))

def get_train_transforms():
    return transforms.Compose([
        # transforms.Lambda(crop_region),

        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

