import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List


class LeftRightDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        resolution,
        label_type,
        trend,
        evlp_setup,  # "homebrew" or "TORex"
        batch_size=16,
        return_label=True,
        grayscale=False,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    ):
        super(LeftRightDataModule, self).__init__()
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(
            os.path.join(data_dir, "labels.csv"), index_col="EVLP_ID"
        )
        self.label_type = label_type
        self.trend = trend
        self.evlp_setup = evlp_setup
        self.resolution = resolution
        self.batch_size = batch_size
        self.return_label = return_label
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.grayscale = grayscale

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.num_labels = 3 if label_type == "multiclass" else 1

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = LeftRightDataset(
                self.data_dir,
                split="train",
                label_type=self.label_type,
                trend=self.trend,
                evlp_setup=self.evlp_setup,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
            self.dataset_val = LeftRightDataset(
                self.data_dir,
                split="val",
                label_type=self.label_type,
                trend=self.trend,
                evlp_setup=self.evlp_setup,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "test" or stage is None:
            self.dataset_test = LeftRightDataset(
                self.data_dir,
                split="test",
                label_type=self.label_type,
                trend=self.trend,
                evlp_setup=self.evlp_setup,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=3
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=3
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=3
        )


class LeftRightDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        resolution,
        label_type,
        trend,
        evlp_setup,
        return_label=True,
        grayscale=False,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    ):
        super(LeftRightDataset, self).__init__()
        assert split in ["train", "val", "test"]

        self.split = split
        self.data_dir = data_dir
        self.split_file = (
            pd.read_csv(
                os.path.join(data_dir, f"{evlp_setup}_trend_random_{split}-list.csv")
            )
            if trend
            else pd.read_csv(
                os.path.join(data_dir, f"{evlp_setup}_no-trend_random_{split}-list.csv")
            )
        )
        self.labels_df = pd.read_csv(
            os.path.join(data_dir, "labels.csv"), index_col="EVLP_ID"
        )
        self.label_type = label_type
        self.resolution = resolution
        self.trend = trend
        self.evlp_setup = evlp_setup
        self.num_labels = 3 if label_type == "multiclass" else 1
        self.return_label = return_label
        self.grayscale = grayscale

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )

    def __len__(self):
        return len(self.split_file) * 2  # Left and right masked images for each EVLP_ID

    def __getitem__(self, item):
        # Get the image tensor (i.e. load it, transform it, turn it into a tensor)
        filename = self.split_file.iloc[item // 2, 0]  # Get one image at a time
        id = int(filename[4:].split("_")[0])

        is_left = item % 2 == 0
        if is_left:  # Get left image first then right image
            mask_val = 2.0
            label = self.labels_df.loc[id, "disposition_left"]
        else:
            mask_val = 1.0
            label = self.labels_df.loc[id, "disposition_right"]

        image_path = os.path.join(self.data_dir, "512p", self.evlp_setup, filename)
        image = Image.open(image_path).convert("RGB")
        if self.grayscale:
            image = image.mean(dim=0, keepdim=True)
        image = np.array(image)
        image = Image.fromarray(image)

        mask_path = os.path.join(
            self.data_dir, "512p_segment-mask", self.evlp_setup, filename
        )
        whole_mask = np.array(Image.open(mask_path))
        mask = whole_mask == mask_val  # Apply left or right mask to the whole mask
        image = image * mask[..., np.newaxis]  # Apply mask to the original image
        image = Image.fromarray(image)
        image = self.transform(image)

        if self.trend:
            image_3hr_path = os.path.join(
                self.data_dir, "512p", self.evlp_setup, filename
            )
            mask_3hr_path = os.path.join(
                self.data_dir, "512p_segment-mask", self.evlp_setup, filename
            )

            image_3hr = Image.open(image_3hr_path).convert("RGB")
            if self.grayscale:
                image_3hr = image_3hr.mean(dim=0, keepdim=True)
            image_3hr = np.array(image_3hr)
            whole_mask_3hr = np.array(Image.open(mask_3hr_path))

            mask_3hr = whole_mask_3hr == mask_val
            image_3hr = image_3hr * mask_3hr[..., np.newaxis]
            image_3hr = Image.fromarray(image_3hr)
            image_3hr.show()
            image_3hr = self.transform(image_3hr)

            image = (image, image_3hr)

        # Get the label tensor
        label = (
            float(self.labels_df.loc[id, "disposition_left"])
            if is_left
            else float(self.labels_df.loc[id, "disposition_right"])
        )
        label = torch.tensor(label)

        if self.return_label:
            return image, label
        else:
            return image

    def get_ids_and_labels(self, trend):  # Used for data inference
        ids, labels_left, labels_right = [], [], []
        for filename in self.split_file.iloc[:, 0]:
            id = int(filename[4:].split("_")[0])
            labels_left.append(self.labels_df.loc[id, "disposition_left"])
            labels_right.append(self.labels_df.loc[id, "disposition_right"])
            if not trend:
                id = filename  # Retain the file name when trend is False (need 1h or 3h info from the filename)
            ids.append(id)

        return ids, labels_left, labels_right

    def get_original_image(self, item):  # Used for saliency mapping
        image_file = self.image_files[item]
        image_path = os.path.join(self.data_dir, "512p", self.evlp_setup, image_file)
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((self.resolution, self.resolution))

        if self.trend:
            image_file_3hr = self.image_files_3hr[item]
            image_path_3hr = os.path.join(self.data_dir, self.split, image_file_3hr)
            image_3hr = Image.open(image_path_3hr).convert("RGB")
            image_3hr_resized = image_3hr.resize((self.resolution, self.resolution))
            image = (image, image_3hr)
            image_resized = (image_resized, image_3hr_resized)

        return image, image_resized


def test_dataloader(data_dir):
    train_dataloader = LeftRightDataModule(
        data_dir="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/EVLP_CXR/left-right_classification/",
        trend=True,
        resolution=512,
        label_type="binary",
        batch_size=2,
        evlp_setup="homebrew",
    )
    train_dataloader.setup(stage="fit")
    train_dataloader = train_dataloader.dataset_train
    # train_dataloader[53]

    dataset = LeftRightDataset(
        data_dir,
        split="train",
        label_type="binary",
        trend=False,
        evlp_setup="homebrew",
        resolution=224,
        return_label=False,
        grayscale=False,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    )

    ids, labels_left, labels_right = dataset.get_ids_and_labels(trend=False)
    print(ids, labels_left, labels_right)


# test_dataloader(
#     "/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/EVLP_CXR/left-right_classification/"
# )
