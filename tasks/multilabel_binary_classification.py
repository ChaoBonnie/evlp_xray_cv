from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class MultiLabelBinaryClassificationTask(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 n_labels: int,
                 lr: float = 1e-3,
                 frozen_feature_extractor: Optional[nn.Module] = None):
        super(MultiLabelBinaryClassificationTask, self).__init__()

        self.model = model
        self.n_labels = n_labels
        self.lr = lr
        self.frozen_feature_extractor = frozen_feature_extractor

        self.train_accuracy = Accuracy(num_classes=self.n_labels, average='none')
        self.val_accuracy = Accuracy(num_classes=self.n_labels, average='none')

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
        y_pred = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        self.log('train_loss', loss)
        self.train_accuracy.update(torch.sigmoid(y_pred), y_true.round().long())

        return loss

    def on_train_epoch_end(self, outputs):
        self.log_per_class_accuracy()

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        self.log('val_loss', loss)
        self.val_accuracy.update(torch.sigmoid(y_pred), y_true.round().long())

    def validation_epoch_end(self, outputs):
        self.log_per_class_accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]

    def log_per_class_accuracy(self):
        assert not self.testing
        if self.training:
            metric = self.train_accuracy
            metric_prefix = 'train'
        else:
            metric = self.val_accuracy
            metric_prefix = 'val'

        class_accuracies = metric.compute()
        self.log(f'{metric_prefix}_accuracy/average', class_accuracies.mean(),
                 on_step=False, on_epoch=True)
        for i, acc in enumerate(class_accuracies):
            self.log(f'{metric_prefix}_accuracy/class={i}', acc,
                     on_step=False, on_epoch=True)
        metric.reset()
