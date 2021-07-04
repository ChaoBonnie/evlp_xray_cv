import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List

conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
              'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
              'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
              'Pleural_Thickening', 'Hernia']


class NIHCXRDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, binary=False, resolution=224, batch_size=32):
        super(NIHCXRDataModule, self).__init__()
        self.data_dir = data_dir
        self.binary = binary
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_labels = 2 if binary else len(conditions)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = NIHCXRDataset(self.data_dir, split='train',
                                               binary=self.binary, resolution=self.resolution)
            self.dataset_val = NIHCXRDataset(self.data_dir, split='val',
                                             binary=self.binary, resolution=self.resolution)
        if stage == "test" or stage is None:
            self.dataset_test = NIHCXRDataset(self.data_dir, split='test',
                                              binary=self.binary, resolution=self.resolution)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=4)


class NIHCXRDataset(Dataset):

    def __init__(self, data_dir, split, binary=False, resolution=224):
        super(NIHCXRDataset, self).__init__()
        assert split in ['train', 'val', 'test']

        self.split = split
        self.data_dir = data_dir
        self.image_files = os.listdir(os.path.join(data_dir, split))
        self.labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'),
                                     index_col='image_id').loc[:, conditions]
        self.binary = binary
        self.resolution = resolution

        # todo: This is where to put data augmentation
        # You can have something like
        # if split == 'train':
        #     self.transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomResizedCrop((self.resolution, self.resolution)),
        #         transforms.RandomRotation(5),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((resolution, resolution)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #     ])
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        # Get the image tensor (i.e. load it, transform it, turn it into a tensor)
        image_file = self.image_files[item]
        image_path = os.path.join(self.data_dir, self.split, image_file)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get the label tensor (and binarize if needed)
        label = self.labels_df.loc[image_file]
        label = torch.from_numpy(label.values)
        if self.binary:
            label = (label == 1).any()
        label = label.float()

        return image, label


def make_labels_file(data_dir):
    source_df = pd.read_csv(os.path.join(data_dir, 'Data_entry_2017_v2020.csv'))
    target_df = []
    for i in range(len(source_df)):
        source_datapoint = source_df.iloc[i]
        target_datapoint = {'image_id': source_datapoint['Image Index'].replace('.png', '.jpg'),
                            'subject_id': source_datapoint['Patient ID']}
        for f in conditions:
            target_datapoint[f] = 0
        findings = source_datapoint['Finding Labels'].split('|')
        for f in findings:
            target_datapoint[f] = 1
        target_df.append(target_datapoint)
    target_df = pd.DataFrame(target_df)
    target_df = target_df.drop(columns='No Finding')
    target_df.to_csv(os.path.join(data_dir, 'labels.csv'), index=False)


def make_split_dirs(data_dir):
    # todo: Make split dirs
    # 1. Make directories inside data_dir called 'train', 'val', and 'test' (os.mkdir(...))
    # 2. Read the train_val_list.txt and test_list.txt
    # 3. Move the images in test_list from the 'images' folder to the 'test' folder
    # 4. Split the image names in train_val_list into a train_list and val_list,
    #    making sure to have different subject_ids in each
    # 5. Move the images in train_list from the 'images' folder to the 'train' folder
    # 6. Move the images in val_list from the 'images' folder to the 'val' folder
    # 7. Delete the empty 'images' folder (but make sure your function works correctly before to not lose all your data!)
    pass
