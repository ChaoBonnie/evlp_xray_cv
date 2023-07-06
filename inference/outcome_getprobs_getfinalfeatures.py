# https://pytorch-lightning.readthedocs.io/en/1.8.0/deploy/production_basic.html
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from torchvision.models.feature_extraction import create_feature_extractor
from scripts.finetune_evlp import load_pretrained
from datasets.evlp_xray_outcome import OutcomeDataModule
from tasks.binary_classification import BinaryClassificationTask
from tasks.multiclass_classification import MulticlassClassificationTask


def get_predicted_probabilities(
    model_path, model_backbone, label_type, csv_name, backbone_saved
):
    num_label = 3 if label_type == "multiclass" else 2

    # if not backbone_saved:
    model, _ = load_pretrained(
        model_backbone=model_backbone,
        finetune_num_labels=num_label,
        trend=True,
        pretrain_num_labels=num_label,
        pretrain_path=None,
        pretrain_was_multilabel=False,
        pretrain_on_all_public_data=False,
        finetune_all=True,
    )
    if label_type == "multiclass":
        model = MulticlassClassificationTask.load_from_checkpoint(
            model_path, model=model  # if not backbone_saved else None
        )
    else:
        model = BinaryClassificationTask.load_from_checkpoint(
            model_path, model=model  # if not backbone_saved else None
        )

    data_loader = OutcomeDataModule(
        data_dir="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/EVLP_CXR/recipient_outcome/Double/Main/",
        resolution=224,
        label_type=label_type,
        trend=True,
        batch_size=1,
        return_label=False,
    )
    data_loader.setup(stage="fit")
    data_loader = data_loader.val_dataloader()

    features = []

    def feature_hook(module, inputs, outputs):
        features.append(inputs[0])

    model.model.classifier.register_forward_hook(feature_hook)

    trainer = pl.Trainer()
    predictions = trainer.predict(model, data_loader)

    predictions = torch.concatenate(predictions, dim=0)
    predictions = predictions.numpy()
    ids, labels = data_loader.dataset.get_ids_and_labels()
    predictions_df = {"id": ids, "label": labels}
    for i in range(predictions.shape[1]):
        predictions_df[f"pred_prob(class={i})"] = predictions[:, i]
    predictions_df = pd.DataFrame(predictions_df)
    predictions_df.to_csv(csv_name.replace(".csv", "_predictions.csv"), index=False)

    features = torch.concatenate(features)
    features = features.numpy()
    all_features_df = pd.DataFrame(features)
    all_features_df.to_csv(csv_name.replace(".csv", "_all-features.csv"), index=False)
    features = PCA(n_components=10).fit_transform(features)
    feature_df = {"id": ids, "label": labels}
    for i in range(features.shape[1]):
        feature_df[f"feature_pca_{i}"] = features[:, i]
    feature_df = pd.DataFrame(feature_df)
    feature_df.to_csv(csv_name.replace(".csv", "_pca-features.csv"), index=False)


# get_predicted_probabilities(
#     model_path="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/saved_models/finetune_evlp/recipient_outcome/recipient_outcome_updatedcsv/trend_rexnet100_NIH_recipient/lightning_logs/version_8419672/checkpoints/epoch=7-step=264.ckpt",
#     model_backbone="rexnet100",
#     label_type="multiclass",
#     csv_name="inference/rexnet100_NIH_recipient.csv",
# )

get_predicted_probabilities(
    model_path="/home/bonnie/Documents/OneDrive_UofT/EVLP_X-ray_Project/evlp_xray_cv/saved_models/finetune_evlp/recipient_outcome/updatedcsv_Adam/trend_resnet50_CADLab_recipient/lightning_logs/version_8808590/checkpoints/epoch=33-step=1122.ckpt",
    model_backbone="resnet50",
    label_type="multiclass",
    csv_name="inference/resnet50_CADLab_recipient.csv",
    backbone_saved=True,
)
