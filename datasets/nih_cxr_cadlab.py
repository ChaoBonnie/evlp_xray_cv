import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List


class NIHCXRCadLabDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, resolution, batch_size=16):
        super(NIHCXRCadLabDataModule, self).__init__()
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_labels = 1

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = NIHCXRCadLabDataset(
                self.data_dir, split="train", resolution=self.resolution
            )
            self.dataset_val = NIHCXRCadLabDataset(
                self.data_dir, split="val", resolution=self.resolution
            )
        if stage == "test" or stage is None:
            self.dataset_test = NIHCXRCadLabDataset(
                self.data_dir, split="test", resolution=self.resolution
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class NIHCXRCadLabDataset(Dataset):
    def __init__(self, data_dir, split, resolution):
        super(NIHCXRCadLabDataset, self).__init__()
        assert split in ["train", "val", "test"]

        self.split = split
        self.data_dir = data_dir
        self.resolution = resolution

        with open(os.path.join(data_dir, split + ".txt"), "r") as f:
            data = f.readlines()
        data = [line.strip().split(" ") for line in data]
        self.images, self.labels = zip(*data)
        self.labels = tuple(float(label) for label in self.labels)

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.CenterCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25),
                    transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Get the image tensor (i.e. load it, transform it, turn it into a tensor)
        image_file = self.images[item]
        image_path = os.path.join(self.data_dir, "images", image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Get the label
        label = self.labels[item]

        return image, label
