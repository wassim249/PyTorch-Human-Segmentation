import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations,data_dir):
        self.df = df
        self.df = df
        self.augmentations = augmentations
        self.data_dir = data_dir
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index :int):
        row = self.df.iloc[index]

        image_path = self._format_path(row.images)
        mask_path = self._format_path(row.masks)

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            image = self.augmentations(image)
            mask = self.augmentations(mask)

        image = image / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

    def split_data(self, val_size=0.2):
        val_size = int(val_size * len(self))
        train_size = len(self) - val_size
        return torch.utils.data.random_split(self, [train_size, val_size])

    def _format_path(self,path : str):
        return path.replace("Human-Segmentation-Dataset-master",os.path.join(self.data_dir))
