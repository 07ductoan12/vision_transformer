import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from torchvision.transforms import v2


class custom_dataset(Dataset):
    def __init__(self, directory: str, transform=None, mode: str = "Train") -> None:
        self.paths = list(glob(directory + "*/*"))

        if transform is not None:
            self.transform = transform
        elif mode == "Train":
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(224, 224)),
                    v2.RandomHorizontalFlip(0.5),
                    v2.RandomVerticalFlip(0.3),
                    # v2.RandomRotation([20, 45]),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif mode == "Valid" or mode == "Test":
            self.transform = v2.Compose(
                [
                    v2.Resize(size=(224, 224)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.classes, self.class_to_idx = self.find_classes(directory)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def find_classes(directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert(mode="RGB")
        class_name = self.paths[index].split("/")[-2]
        return self.transform(image), self.class_to_idx[class_name]


def load_dataset(train_dir: str, valid_dir: str, num_workers: int, batch_size: int):
    train_dataset = custom_dataset(train_dir, mode="Train")
    valid_dataset = custom_dataset(valid_dir, mode="Valid")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, valid_dataloader, len(train_dataset.classes)
