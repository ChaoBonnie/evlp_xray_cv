import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision.models import resnet50, densenet121
from datasets.nih_cxr import NIHCXRDataModule
from datasets.nih_cxr_cadlab import NIHCXRCadLabDataModule
from tasks.binary_classification import BinaryClassificationTask
from tasks.multilabel_binary_classification import MultiLabelBinaryClassificationTask


def main(data_dir, save_dir, binary, model_backbone,
         batch_size=64, max_epochs=None, use_cadlab=False, debug=False):
    if use_cadlab:
        assert binary
        data = NIHCXRCadLabDataModule(data_dir, batch_size=batch_size)
    else:
        data = NIHCXRDataModule(data_dir, binary=binary, batch_size=batch_size)

    # Load the ImageNet pre-trained model backbone and change the number of units at the output
    if model_backbone == 'resnet50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=data.num_labels)
    if model_backbone == 'densenet121':
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=data.num_labels)
    else:
        raise ValueError(f'Unknown model backbone: {model_backbone}')

    if binary:
        task = BinaryClassificationTask(model=model)
    else:
        task = MultiLabelBinaryClassificationTask(model=model, n_labels=data.num_labels)

    # todo: add an EarlyStopping callback
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=max_epochs,
        auto_select_gpus=True,
        overfit_batches=1 if debug else 0,
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=0
    )
    trainer.fit(model=task, datamodule=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain on NIH CXR dataset.')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='Directory containing NIH CXR data.')
    parser.add_argument('--name', required=True, type=str,
                        help='Custom location for saving checkpoints and logs.')
    parser.add_argument('--model_backbone', default='densenet121', type=str,
                        choices=['resnet50', 'densenet121'],
                        help='Base model architecture to use.')
    parser.add_argument('--binary', action='store_true',
                        help='Perform binary finding/no-finding classification (rather than per-class prediction).')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training.')
    parser.add_argument('--max_epochs', default=100, type=int,
                        help='Maximum number of epochs to train for.')
    parser.add_argument('--cadlab_dataset', action='store_true',
                        help='Use the subset of images and labels from the 2020 CAD Lab paper.')
    parser.add_argument('--debug', action='store_true',
                        help='Overfit to a batch in order to debug the code/model/dataset.')
    args = parser.parse_args()

    save_dir = f'saved_models/pretrain_nihcxr/{args.name}'

    main(data_dir=args.data_dir, save_dir=save_dir, binary=args.binary, model_backbone=args.model_backbone,
         batch_size=args.batch_size, max_epochs=args.max_epochs, use_cadlab=args.cadlab_dataset, debug=args.debug)
