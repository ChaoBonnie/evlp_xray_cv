import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class BinaryClassificationTask(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3):
        super(BinaryClassificationTask, self).__init__()

        self.model = model
        self.lr = lr

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy(compute_on_step=False)      # Only compute on the entire validation set

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        self.log('train_loss', loss)
        self.train_accuracy(y_pred, y_true)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y_true)

        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy(y_pred, y_true)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
