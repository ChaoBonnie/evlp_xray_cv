from typing import Optional
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix


class BinaryClassificationTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        frozen_feature_extractor: Optional[nn.Module] = None,
    ):
        super(BinaryClassificationTask, self).__init__()

        self.model = model
        self.lr = lr
        self.frozen_feature_extractor = frozen_feature_extractor

        self.train_accuracy = Accuracy()
        self.train_cm = ConfusionMatrix(num_classes=2)
        self.val_accuracy = Accuracy()
        self.val_cm = ConfusionMatrix(num_classes=2)

    def forward(self, x):
        if self.frozen_feature_extractor is None:
            y_pred = self.model(x)
        else:
            self.frozen_feature_extractor.eval()
            with torch.no_grad():
                x = self.frozen_feature_extractor(x)
            y_pred = self.model(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x).reshape(-1)

        loss = ops.sigmoid_focal_loss(y_pred, y_true)

        self.log("train_loss", loss)
        self.update_logs(y_pred, y_true)

        return loss

    def training_epoch_end(self, outputs):
        self.make_logs()

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x).reshape(-1)

        loss = ops.sigmoid_focal_loss(y_pred, y_true)

        self.log("val_loss", loss)
        self.update_logs(y_pred, y_true)

    def validation_epoch_end(self, outputs):
        self.make_logs()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=0.0001, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def update_logs(self, y_pred, y_true):
        assert not self.testing
        if self.training:
            acc, cm = self.train_accuracy, self.train_cm
        else:
            acc, cm = self.val_accuracy, self.val_cm

        y_pred, y_true = torch.sigmoid(y_pred), y_true.round().long()

        acc.update(y_pred, y_true)
        cm.update(y_pred, y_true)

    def make_logs(self):
        assert not self.testing
        if self.training:
            acc, cm = self.train_accuracy, self.train_cm
            metric_prefix = "train"
        else:
            acc, cm = self.val_accuracy, self.val_cm
            metric_prefix = "val"

        self.log(f"{metric_prefix}_accuracy", acc.compute())
        self.log_confusion_matrix(f"{metric_prefix}_cm", cm.compute())

        acc.reset()
        cm.reset()

    def log_confusion_matrix(self, name, cm):
        df_cm = pd.DataFrame(cm.numpy(), index=[0, 1], columns=[0, 1])
        df_cm = df_cm.rename_axis(index="True", columns="Predicted")
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(df_cm, annot=True)
        fig = ax.get_figure()
        plt.close(fig)
        self.logger.experiment.add_figure(name, fig, self.global_step)
