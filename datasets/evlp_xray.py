import os
import shutil
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List


class EVLPDataModule(pl.LightningDataModule):

    def __init__(self, data_dir,
                 binarize=False, discretize=False,
                 aggregate_regions=False, aggregate_labels=False,
                 resolution=224, batch_size=16):
        super(EVLPDataModule, self).__init__()
        self.data_dir = data_dir
        self.binarize = binarize
        self.discretize = discretize
        self.aggregate_regions = aggregate_regions
        self.aggregate_labels = aggregate_labels
        self.resolution = resolution
        self.batch_size = batch_size

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        if aggregate_labels:
            self.num_labels = 1
        else:
            self.num_labels = len(self.labels_df.columns)
            # todo: comment out the code below once we start using the raw, unaggregated dataframe
            # if aggregate_spatial:
            #     self.num_labels = self.num_labels // 6      # We are aggregating across 6 lung regions

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = EVLPDataset(self.data_dir,
                                             split='train',
                                             binarize=self.binarize,
                                             discretize=self.discretize,
                                             aggregate_regions=self.aggregate_regions,
                                             aggregate_labels=self.aggregate_labels,
                                             resolution=self.resolution)
            self.dataset_val = EVLPDataset(self.data_dir,
                                           split='val',
                                           binarize=self.binarize,
                                           discretize=self.discretize,
                                           aggregate_regions=self.aggregate_regions,
                                           aggregate_labels=self.aggregate_labels,
                                           resolution=self.resolution)
        if stage == "test" or stage is None:
            self.dataset_test = EVLPDataset(self.data_dir,
                                            split='test',
                                            binarize=self.binarize,
                                            discretize=self.discretize,
                                            aggregate_regions=self.aggregate_regions,
                                            aggregate_labels=self.aggregate_labels,
                                            resolution=self.resolution)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4)


class EVLPDataset(Dataset):

    def __init__(self, data_dir, split,
                 binarize=False, discretize=False,
                 aggregate_regions=False, aggregate_labels=False,
                 resolution=224):
        super(EVLPDataset, self).__init__()
        assert split in ['train', 'val', 'test']
        assert not (binarize and discretize)
        if aggregate_labels:
            assert aggregate_regions and binarize

        self.split = split
        self.data_dir = data_dir
        self.image_files = os.listdir(os.path.join(data_dir, split))
        self.labels_df = pd.read_excel(os.path.join(data_dir, 'labels.xlsx'), index_col='EVLP_ID')
        self.binarize = binarize
        self.discretize = discretize  # todo: use me for something
        self.aggregate_regions = aggregate_regions  # todo: use me for something
        self.aggregate_labels = aggregate_labels  # todo: think about how to use this in the case of regression and multi-class classification
        self.resolution = resolution

        if aggregate_labels:
            self.num_labels = 1
        else:
            self.num_labels = len(self.labels_df.columns)
            # todo: comment out the code below once we start using the raw, unaggregated dataframe
            # if aggregate_spatial:
            #     self.num_labels = self.num_labels // 6      # We are aggregating across 6 lung regions

        # todo: This is where to put data augmentation
        if split == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop((self.resolution, self.resolution), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.Resize((resolution, resolution)),
                transforms.RandomRotation(10),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), #Placeholders - remember to update
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        # self.transform = transforms.Compose([
        #     transforms.Resize((resolution, resolution)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        # Get the image tensor (i.e. load it, transform it, turn it into a tensor)
        image_file = self.image_files[item]
        image_path = os.path.join(self.data_dir, self.split, image_file)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get the label tensor
        # todo: apply the correct aggregations/discretizations to the labels when I start using the raw data.
        #  For now, I'm using aggregated regression data as input to this class, which makes it less flexible.
        evlp_id = image_file.split('.')[0]  # Remove the file extension
        label = self.labels_df.loc[evlp_id]
        label = torch.from_numpy(label.values)
        label = label.float()
        if self.binarize:
            label = self.binarize_label(label)

        return image, label

    def discretize_label(self, label):
        # todo: not used now. Worry about how to do this with the raw data
        pass

    def binarize_label(self, label):
        label = (label > 1)
        if self.aggregate_labels:
            label = label.any()
        label = label.float()
        return label
