import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ThymomaDataset(Dataset):
    def __init__(
        self,
        image_root,
        label_json,
        is_train=True,
        max_slices=64,
        target_slices=64,
        image_size=224,
    ):
        self.image_root = image_root
        self.max_slices = max_slices
        self.target_slices = target_slices
        self.is_train = is_train

        with open(label_json, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.patient_ids = list(self.labels.keys())

        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self.augment_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=30, fill=0),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            ),
        ])
        self.post_tensor_transform = T.RandomErasing(
            p=0.3,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0,
        )

    def __len__(self):
        return len(self.patient_ids)

    @staticmethod
    def _slice_number(filename):
        try:
            return int(os.path.splitext(filename)[0].lstrip("0") or "0")
        except ValueError:
            return 0

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = int(self.labels[patient_id])
        patient_image_dir = os.path.join(self.image_root, patient_id)
        image_files = sorted(
            (
                filename
                for filename in os.listdir(patient_image_dir)
                if filename.lower().endswith((".png", ".jpg", ".jpeg"))
            ),
            key=self._slice_number,
        )
        image_paths = [os.path.join(patient_image_dir, filename) for filename in image_files]
        if not image_paths:
            raise FileNotFoundError(
                f"No image files found for patient {patient_id}: {patient_image_dir}"
            )

        total_slices = len(image_paths)
        if total_slices >= self.target_slices:
            interval = total_slices / self.target_slices
            indices = [int(i * interval) for i in range(self.target_slices)]
        else:
            indices = np.linspace(
                0,
                total_slices - 1,
                self.target_slices,
                dtype=int,
            ).tolist()

        images = [Image.open(image_paths[i]).convert("RGB") for i in indices]
        if self.is_train:
            images = [self.augment_transforms(image) for image in images]

        image_tensors = [self.base_transform(image) for image in images]
        if self.is_train:
            image_tensors = [self.post_tensor_transform(image) for image in image_tensors]

        if len(image_tensors) < self.max_slices:
            pad_image = torch.zeros_like(image_tensors[0])
            image_tensors.extend(
                [pad_image] * (self.max_slices - len(image_tensors))
            )

        image_tensor = torch.stack(image_tensors[:self.max_slices])
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor
