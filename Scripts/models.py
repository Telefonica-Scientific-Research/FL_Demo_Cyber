"""
models.py
=========
DNN model for network intrusion detection (multi-class classification).
Architecture mirrors those in the original Edge-IIoTset paper:
feed-forward layers with batch normalisation and dropout regularisation.
"""

import torch
import torch.nn as nn


class IntrusionDetectionDNN(nn.Module):
    """
    Multi-layer feed-forward DNN for network intrusion detection.

    Parameters
    ----------
    input_dim   : number of input features (after preprocessing)
    num_classes : number of output classes (attack types + Normal)
    hidden_dims : tuple of hidden layer widths
    dropout     : dropout probability applied after each hidden activation
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, num_classes: int) -> IntrusionDetectionDNN:
    """Convenience factory function."""
    return IntrusionDetectionDNN(input_dim, num_classes)
