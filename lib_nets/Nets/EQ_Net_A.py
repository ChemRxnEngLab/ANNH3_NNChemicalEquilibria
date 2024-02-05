import torch.nn as nn
import torch
from typing import Optional


class NeuralNetwork(nn.Module):
    # Initalisierung der Netzwerk layers
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        hidden3_size: int,
        output_size: int,
        mean_in: Optional[torch.Tensor] = None,
        mean_out: Optional[torch.Tensor] = None,
        std_in: Optional[torch.Tensor] = None,
        std_out: Optional[torch.Tensor] = None,
    ):
        super().__init__()  # Referenz zur Base Class (nn.Module)
        # Kaskade der Layer
        self.linear_afunc_stack = nn.Sequential(
            # nn.BatchNorm1d(input_size), # Normalisierung, damit Inputdaten gleiche Größenordnung haben
            nn.Linear(input_size, hidden1_size),
            nn.GELU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.GELU(),
            nn.Linear(hidden2_size, hidden3_size),
            nn.GELU(),
            nn.Linear(hidden3_size, output_size),
        )

        self.mean_in = nn.Parameter(
            torch.zeros(input_size) if mean_in is None else mean_in,
            requires_grad=False,
        )
        self.std_in = nn.Parameter(
            torch.ones(input_size) if std_in is None else std_in,
            requires_grad=False,
        )
        self.mean_out = nn.Parameter(
            torch.zeros(output_size) if mean_out is None else mean_out,
            requires_grad=False,
        )
        self.std_out = nn.Parameter(
            torch.ones(output_size) if std_out is None else std_out,
            requires_grad=False,
        )

    # Implementierung der Operationen auf Input Daten
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = (x - self.mean_in) / self.std_in
        out_t = self.linear_afunc_stack(x_t)
        out = out_t * self.std_out + self.mean_out
        return out
