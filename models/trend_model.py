from typing import Tuple
import torch
from torch import nn


class TrendModel(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, num_feats: int, num_outputs: int
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.num_outputs = num_outputs

        # Either a linear layer or an MLP
        self.classifier = nn.Linear(in_features=num_feats * 2, out_features=num_outputs)

    def forward(self, x_1hr_3hr: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size = x_1hr_3hr[0].shape[0]

        # Get the CNN features
        x_stacked = torch.cat(
            x_1hr_3hr, dim=0
        )  # (batch, 3, x, y), (batch, 3, x, y) -> (batch*2, 3, x, y)
        feats_stacked = self.feature_extractor(x_stacked)
        feats_stacked = feats_stacked.flatten(1)
        # pass through your CNN (1 CNN processes 1hr and 3hr independently). (batch*2, 3, x, y) -> (batch*2, num_features)

        # Get the trend features
        feats_1hr, feats_3hr = torch.split(
            feats_stacked, batch_size, dim=0
        )  # split into 2 tensors (1hr and 3hr)
        feats_trend = torch.cat(
            [feats_1hr, feats_3hr], dim=1
        )  # (batch, num_features), (batch, num_features) -> (batch, num_features*2)

        # Get the predictions
        y_pred = self.classifier(
            feats_trend
        )  # pass through your classifier (1 classifier processes 1hr and 3hr together). (batch, num_features*2) -> (batch, num_classes)

        return y_pred
