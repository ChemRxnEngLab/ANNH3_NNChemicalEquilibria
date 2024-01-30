import torch.nn as nn
import torch
from typing import Optional
from collections import OrderedDict


class NeuralNetwork(nn.Module):
    # Initalisierung der Netzwerk layers
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        hidden3_size: int,
        output_size: int,
    ):
        super().__init__()  # Referenz zur Base Class (nn.Module)
        # Kaskade der Layer
        layers: OrderedDict[str, nn.Module] = OrderedDict(
            [
                ("BatchNorm", nn.BatchNorm1d(input_size, momentum=0.2, affine=False)),
                ("Linear1", nn.Linear(input_size, hidden1_size)),
                ("GELU1", nn.GELU()),
                ("Linear2", nn.Linear(hidden1_size, hidden2_size)),
                ("GELU2", nn.GELU()),
                ("Linear3", nn.Linear(hidden2_size, hidden3_size)),
                ("GELU3", nn.GELU()),
                ("Linear4", nn.Linear(hidden3_size, output_size)),
            ]
        )
        self.linear_afunc_stack = nn.Sequential(layers)
        self.norm = self.linear_afunc_stack[0]

    # Implementierung der Operationen auf Input Daten
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_afunc_stack(x)
        return out
