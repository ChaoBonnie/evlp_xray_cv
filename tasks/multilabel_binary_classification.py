import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class MultiClassificationTask(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 n_classes: int,
                 lr: float = 1e-3):
        super(MultiClassificationTask, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.lr = lr

        self.train_accuracy = Accuracy(num_classes=self.n_classes, average='none')
        self.val_accuracy = Accuracy(num_classes=self.n_classes, average='none')

    def forward(self, x):
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
        class_accuracies = self.train_accuracy.compute()
        self.log('train_accuracy/average', class_accuracies.mean())
        for i, acc in enumerate(class_accuracies):
            self.log(f'train_accuracy/class={i}', acc)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy.update(torch.sigmoid(y_pred), y_true.round().long())

    def validation_epoch_end(self, outputs):
        class_accuracies = self.val_accuracy.compute()
        self.log('val_accuracy/average', class_accuracies.mean())
        for i, acc in enumerate(class_accuracies):
            self.log(f'val_accuracy/class={i}', acc)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]