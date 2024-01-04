import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


class LungCTLoader(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(self.data_dir+img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def custom_collate(batch):
    # No need to convert to tensor again, as ToTensor() is applied in LungCTLoader
    return torch.utils.data.dataloader.default_collate(batch)

class LungCTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, transform=None):
        super(LungCTDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    def setup(self, stage=None):
        # Assign train/val/test datasets
        if stage == "fit" or stage is None:
            self.dataset = LungCTLoader(data_dir=self.data_dir, transform=self.transform)
            self.dataset_size = len(self.dataset)
            train_size = int(0.8 * self.dataset_size)
            val_size = self.dataset_size - train_size
            self.lungct_train, self.lungct_val = random_split(self.dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.lungct_test = LungCTLoader(data_dir=self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.lungct_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def val_dataloader(self):
        return DataLoader(self.lungct_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=custom_collate)

    def test_dataloader(self):
        return DataLoader(self.lungct_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=custom_collate)