import os
import shutil
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List
import random

conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
              'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
              'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
              'Pleural_Thickening', 'Hernia']


class NIHCXRDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, binary=False, resolution=224, batch_size=16):
        super(NIHCXRDataModule, self).__init__()
        self.data_dir = data_dir
        self.binary = binary
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_labels = 1 if binary else len(conditions)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

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
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(int(resolution * 1.143)),  # 256 when resolution = 224
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(resolution * 1.143)),  # 256 when resolution = 224
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
        target_datapoint = {'image_id': source_datapoint['Image Index'],
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


def make_split_dirs(image_dir, split_list_dir, val_size=0.2):
    """
Note: assumes that images have been resized and saved as jpg's (rather than png's)
    """
    train_dir = image_dir + '/train'
    val_dir = image_dir + '/val'
    test_dir = image_dir + '/test'

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    with open(split_list_dir + '/test_list.txt') as f:
        test_filenames = f.readlines()
    for filename in test_filenames:
        filename = filename.strip().replace('.png', '.jpg')
        shutil.move(image_dir + '/' + filename, test_dir + '/' + filename)

    with open(split_list_dir + '/train_val_list.txt') as f:
        train_val_filenames = f.readlines()
    subject_ids = [filename.split('_')[0] for filename in train_val_filenames]
    subject_ids = list(set(subject_ids))
    random.seed(35)
    random.shuffle(subject_ids)
    val_subject_ids = set(subject_ids[:int(len(subject_ids)*val_size)])
    train_subject_ids = set(subject_ids[int(len(subject_ids)*val_size):])
    train_filenames = [filename for filename in train_val_filenames if filename.split('_')[0] in train_subject_ids]
    val_filenames = [filename for filename in train_val_filenames if filename.split('_')[0] in val_subject_ids]

    for filename in train_filenames:
        filename = filename.strip().replace('.png', '.jpg')
        shutil.move(image_dir + '/' + filename, train_dir + '/' + filename)
    for filename in val_filenames:
        filename = filename.strip().replace('.png', '.jpg')
        shutil.move(image_dir + '/' + filename, val_dir + '/' + filename)
make_labels_file('C:/Users/chaob/OneDrive - University of Toronto/Documents/EVLP X-ray Imaging Project')