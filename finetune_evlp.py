import argparse
import timm
from copy import deepcopy
from typing import Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision.models import (
    densenet121,
    efficientnet_b2,
    efficientnet_b3,
    resnet50,
    resnext50_32x4d,
    rexnet_100,
)
from datasets.evlp_xray_outcome import OutcomeDataModule
from tasks.binary_classification import BinaryClassificationTask
from tasks.multilabel_binary_classification import MultiLabelBinaryClassificationTask
from tasks.multiclass_classification import MulticlassClassificationTask
from models.trend_model import TrendModel


def main(
    data_dir,
    save_dir,
    model_backbone,
    classification,
    pretrain_path=None,
    finetune_all=True,
    pretrain_was_multilabel=False,
    pretrain_num_labels=1,
    max_epochs=None,
    debug=False,
):
    # Asserts for valid argument combinations
    if not pretrain_was_multilabel:
        assert pretrain_num_labels == 1

    # Load the pre-trained model
    model, frozen_feature_extractor, resolution = load_pretrained(
        model_backbone=model_backbone,
        pretrain_path=pretrain_path,
        pretrain_was_multilabel=pretrain_was_multilabel,
        pretrain_num_labels=pretrain_num_labels,
        finetune_num_labels=data.num_labels,
        finetune_all=finetune_all,
    )

    # Load your data
    data = OutcomeDataModule(
        data_dir,
        resolution=resolution,
        batch_size=32,
    )

    # Create an instance of the task we want to be training on

    """
    Tasks used for RLS classification:
    
    if binarize and aggregate_labels and aggregate_regions:
        task = BinaryClassificationTask(
            model=model, frozen_feature_extractor=frozen_feature_extractor
        )
    elif binarize and not aggregate_labels:
        task = MultiLabelBinaryClassificationTask(
            model=model,
            n_labels=data.num_labels,
            frozen_feature_extractor=frozen_feature_extractor,
        )
    else:
        raise ValueError(
            f"Currently unsupported training combination for:"
            f'["binarize", "discretize", "aggregate_regions", "aggregate_labels"]'
        )
    """
    if classification == "binary":
        task = BinaryClassificationTask(
            model=model, frozen_feature_extractor=frozen_feature_extractor
        )
    elif classification == "multiclass":
        task = MulticlassClassificationTask(
            model=model, frozen_feature_extractor=frozen_feature_extractor
        )

    # todo: add an EarlyStopping callback
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    # Train the model
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=max_epochs,
        auto_select_gpus=True,
        overfit_batches=1 if debug else 0,
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model=task, datamodule=data)


def load_pretrained(
    model_backbone: str,
    pretrain_path: Optional[str],
    pretrain_was_multilabel: bool,
    pretrain_num_labels: int,
    finetune_num_labels: int,
    finetune_all: bool,
    trend: bool = True,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    # Step 1: recreate the pre-trained model backbone, so that we can load its weights
    if model_backbone == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "resnext50_32x4d":
        model = resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "rexnet_100":
        model = timm.create_model("rexnet_100", pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "densenet121":
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "efficientnet_b2":
        model = efficientnet_b2(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=pretrain_num_labels
        )
    elif model_backbone == "efficientnet_b3":
        model = efficientnet_b3(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=pretrain_num_labels
        )
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    # Step 2: load the model from its checkpoint
    if pretrain_path is not None:
        if pretrain_was_multilabel:
            task = MultiLabelBinaryClassificationTask.load_from_checkpoint(
                pretrain_path, model=model, n_classes=pretrain_num_labels
            )
            model = task.model
        else:
            task = BinaryClassificationTask.load_from_checkpoint(
                pretrain_path, model=model
            )
            model = task.model

    # Step 3: Swap out the last layer for our number of labels
    resolution = 224
    if (model_backbone == "resnet50") | (model_backbone == "resnext50_32x4d"):
        if not trend:
            print("old num-classes: ", model.fc)  # 1000
            model.fc = nn.Linear(
                in_features=model.fc.in_features, out_features=finetune_num_labels
            )
            print("new num-classes: ", model.fc)  # 3
        else:
            num_feats = model.fc.in_features
            model.fc = nn.Identity()
    elif (
        (model_backbone == "densenet121")
        | (model_backbone == "efficientnet_b2")
        | (model_backbone == "efficientnet_b3")
    ):
        if not trend:
            print("old num-classes: ", model.classifier)  # 1000
            model.classifier = nn.Linear(
                in_features=model.classifier.in_features,
                out_features=finetune_num_labels,
            )
            print("new num-classes: ", model.classifier)  # 3
        else:
            num_feats = model.classifier.in_features
            model.classifier = nn.Identity()
    # elif args.model == "rexnet_100":
    #     if not trend:
    #         print("old num-classes: ", model.head)  # 1000
    #         model.reset_classifier(num_classes=finetune_num_labels) ### Find the last layer of rexnet_100!
    #         print("new num-classes: ", model.head)  # 3
    #     else:
    #         num_feats = model.head.in_features
    #         model.head = nn.Identity()
    else:
        raise ValueError(f"Unknown model backbone: {model_backbone}")

    # Step 4: Set the trainable parameters
    if finetune_all:
        frozen_feature_extractor = None
    else:
        if model_backbone == "resnet50":
            frozen_feature_extractor = deepcopy(model)
            frozen_feature_extractor.fc = nn.Identity()
            model = model.fc
        else:
            raise ValueError(f"Unknown model backbone: {model_backbone}")

    if trend:
        model = TrendModel(
            feature_extractor=model,
            num_feats=num_feats,
            num_outputs=finetune_num_labels,
        )

    return model, frozen_feature_extractor, resolution


if __name__ == "__main__":
    # todo: some of these arguments might have to change for EVLP fine-tuning (e.g. you need to specify a pre-trained model path)
    parser = argparse.ArgumentParser(description="Fine-tune on EVLP dataset.")
    parser.add_argument(
        "--data_dir", required=True, type=str, help="Directory containing EVLP data."
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Custom location for saving checkpoints and logs.",
    )
    parser.add_argument(
        "--model_backbone",
        default="resnet50",
        type=str,
        choices=[
            "resnet50",
            "resnext50_32x4d",
            "rexnet_100",
            "densenet121",
            "efficientnet_b2",
            "efficientnet_b3",
        ],
        help="Base model architecture to use.",
    )
    parser.add_argument(
        "--classification",
        required=True,
        type=str,
        help="Binary or multiclass classification.",
    )
    parser.add_argument(
        "--pretrain_path", type=str, help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--finetune_head",
        action="store_true",
        help="Fine-tune just the model head (last little bit).",
    )
    parser.add_argument(
        "--pretrain_was_multilabel",
        action="store_true",
        help="Pre-trained model was trained on multi-label task.",
    )
    parser.add_argument(
        "--pretrain_num_labels",
        type=int,
        default=1,
        help="Number of labels used in the pre-training task.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Turn label scores into binary finding/no-finding.",
    )
    parser.add_argument(
        "--discretize",
        action="store_true",
        help="Turn label scores into some discrete quantiles (multi-class classification).",
    )
    parser.add_argument(
        "--aggregate_regions",
        action="store_true",
        help="Combine labels across regions.",
    )
    parser.add_argument(
        "--aggregate_labels",
        action="store_true",
        help="Combine labels across radiological finding types.",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Image resolution (depending on the model backbone).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Overfit to a batch in order to debug the code/model/dataset.",
    )
    args = parser.parse_args()

    save_dir = f"saved_models/finetune_evlp/{args.name}"

    main(
        data_dir=args.data_dir,
        save_dir=save_dir,
        model_backbone=args.model_backbone,
        classification=args.classification,
        pretrain_path=args.pretrain_path,
        finetune_all=not args.finetune_head,
        pretrain_was_multilabel=args.pretrain_was_multilabel,
        pretrain_num_labels=args.pretrain_num_labels,
        max_epochs=args.max_epochs,
        resolution=args.resolution,
        debug=args.debug,
    )
