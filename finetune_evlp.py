import argparse
from copy import deepcopy
from typing import Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision.models import resnet50
from datasets.evlp_xray import EVLPDataModule
from tasks.binary_classification import BinaryClassificationTask
from tasks.multilabel_binary_classification import MultiLabelBinaryClassificationTask


def main(data_dir, save_dir, model_backbone,
         pretrain_path=None, finetune_all=True,
         pretrain_was_multilabel=False, pretrain_num_labels=1,
         binarize=False, discretize=False,
         aggregate_regions=False, aggregate_labels=False,
         max_epochs=None,
         debug=False):
    # Asserts for valid argument combinations
    if not pretrain_was_multilabel:
        assert pretrain_num_labels == 1

    # Load your data
    data = EVLPDataModule(data_dir, binarize=binarize, discretize=discretize,
                          aggregate_regions=aggregate_regions, aggregate_labels=aggregate_labels,
                          batch_size=1)  # todo: increase batch size once larger dataset is acquired

    # Load the pre-trained model
    model, frozen_feature_extractor = load_pretrained(model_backbone=model_backbone,
                                                      pretrain_path=pretrain_path,
                                                      pretrain_was_multilabel=pretrain_was_multilabel,
                                                      pretrain_num_labels=pretrain_num_labels,
                                                      finetune_num_labels=data.num_labels,
                                                      finetune_all=finetune_all)

    # Create an instance of the task we want to be training on
    # todo: update these, since we probably don't want to just support binary/multilabel binary for EVLP
    if binarize and aggregate_labels and aggregate_regions:
        task = BinaryClassificationTask(model=model,
                                        frozen_feature_extractor=frozen_feature_extractor)
    elif binarize and not aggregate_labels:
        task = MultiLabelBinaryClassificationTask(model=model,
                                                  n_labels=data.num_labels,
                                                  frozen_feature_extractor=frozen_feature_extractor)
    else:
        raise ValueError(f'Currently unsupported training combination for:'
                         f'["binarize", "discretize", "aggregate_regions", "aggregate_labels"]')

    # todo: add an EarlyStopping callback
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    # Train the model
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=max_epochs,
        auto_select_gpus=True,
        overfit_batches=1 if debug else 0,
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=0
    )
    trainer.fit(model=task, datamodule=data)


def load_pretrained(model_backbone: str,
                    pretrain_path: Optional[str],
                    pretrain_was_multilabel: bool,
                    pretrain_num_labels: int,
                    finetune_num_labels: int,
                    finetune_all: bool) -> Tuple[nn.Module, Optional[nn.Module]]:
    # Step 1: recreate the pre-trained model backbone, so that we can load its weights
    if model_backbone == 'resnet50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=pretrain_num_labels)
    else:
        raise ValueError(f'Unknown model backbone: {model_backbone}')

    # Step 2: load the model from its checkpoint
    if pretrain_path is not None:
        if pretrain_was_multilabel:
            task = MultiLabelBinaryClassificationTask.load_from_checkpoint(pretrain_path,
                                                                           model=model, n_classes=pretrain_num_labels)
            model = task.model
        else:
            task = BinaryClassificationTask.load_from_checkpoint(pretrain_path,
                                                                 model=model)
            model = task.model

    # Step 3: Swap out the last layer for our number of labels
    if model_backbone == 'resnet50':
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=finetune_num_labels)
    else:
        raise ValueError(f'Unknown model backbone: {model_backbone}')

    # Step 4: Set the trainable parameters
    if finetune_all:
        frozen_feature_extractor = None
    else:
        if model_backbone == 'resnet50':
            frozen_feature_extractor = deepcopy(model)
            frozen_feature_extractor.fc = nn.Identity()
            model = model.fc
        else:
            raise ValueError(f'Unknown model backbone: {model_backbone}')

    return model, frozen_feature_extractor


if __name__ == '__main__':
    # todo: some of these arguments might have to change for EVLP fine-tuning (e.g. you need to specify a pre-trained model path)
    parser = argparse.ArgumentParser(description='Fine-tune on EVLP dataset.')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='Directory containing EVLP data.')
    parser.add_argument('--name', required=True, type=str,
                        help='Custom location for saving checkpoints and logs.')
    parser.add_argument('--model_backbone', default='resnet50', type=str,
                        choices=['resnet50'],
                        help='Base model architecture to use.')
    parser.add_argument('--pretrain_path', type=str,
                        help='Path to the pre-trained model.')
    parser.add_argument('--finetune_head', action='store_true',
                        help='Fine-tune just the model head (last little bit).')
    parser.add_argument('--pretrain_was_multilabel', action='store_true',
                        help='Pre-trained model was trained on multi-label task.')
    parser.add_argument('--pretrain_num_labels', type=int, default=1,
                        help='Number of labels used in the pre-training task.')
    parser.add_argument('--binarize', action='store_true',
                        help='Turn label scores into binary finding/no-finding.')
    parser.add_argument('--discretize', action='store_true',
                        help='Turn label scores into some discrete quantiles (multi-class classification).')
    parser.add_argument('--aggregate_regions', action='store_true',
                        help='Combine labels across regions.')
    parser.add_argument('--aggregate_labels', action='store_true',
                        help='Combine labels across radiological finding types.')
    parser.add_argument('--max_epochs', default=100, type=int,
                        help='Maximum number of epochs to train for.')
    parser.add_argument('--debug', action='store_true',
                        help='Overfit to a batch in order to debug the code/model/dataset.')
    args = parser.parse_args()

    save_dir = f'saved_models/finetune_evlp/{args.name}'

    main(data_dir=args.data_dir, save_dir=save_dir, model_backbone=args.model_backbone,
         pretrain_path=args.pretrain_path, finetune_all=not args.finetune_head,
         pretrain_was_multilabel=args.pretrain_was_multilabel, pretrain_num_labels=args.pretrain_num_labels,
         binarize=args.binarize, discretize=args.discretize,
         aggregate_regions=args.aggregate_regions, aggregate_labels=args.aggregate_labels,
         max_epochs=args.max_epochs, debug=args.debug)
