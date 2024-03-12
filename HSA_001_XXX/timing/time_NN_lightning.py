# Importe / Bibliotheken
from pathlib import Path
import numpy as np
import torch
import lightning.pytorch as pl
from typing import Any
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.constants import R
import timeit
import sys

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)


from Nets.EQ_Net_A import NeuralNetwork
from GGW_calc.GGW import GGW
from ICIW_Plots import make_square_ax

import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Parameter
num = 10_000_000  # Anzahl der Werte im Vektor

np.random.seed(42)

T = np.random.uniform(408.15, 1273.15, num)  # K Temperatur
p = np.random.uniform(1, 500, num)  # bar Druck

T = torch.tensor(T)
p = torch.tensor(p)

# Stofffmengen zu Reaktionsbeginn
n_ges_0 = 1  # mol Gesamtstoffmenge zum Reaktionsbeginn
x_0 = np.random.dirichlet((1, 1, 1), num)  # 1 Stoffmengenanteile zu Reaktionsbeginn
n_H2_0 = x_0[:, 0] * n_ges_0  # mol Stoffmenge H2 Start
n_N2_0 = x_0[:, 1] * n_ges_0  # mol Stoffmenge N2 Start
n_NH3_0 = x_0[:, 2] * n_ges_0  # mol Stoffmenge NH3 Start

n_H2_0 = torch.tensor(n_H2_0)
n_N2_0 = torch.tensor(n_N2_0)
n_NH3_0 = torch.tensor(n_NH3_0)

X = torch.stack((T, p, n_H2_0, n_N2_0, n_NH3_0), dim=1)

##load the network
torch.set_default_dtype(torch.float64)

net_file = Path.cwd() / "models" / "torch" / "NH3_net_LU.pt"


# load model
class EqNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        hidden3_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.net = NeuralNetwork(
            input_size,
            hidden1_size,
            hidden2_size,
            hidden3_size,
            output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, y = batch
        pred = self.net(X)

        mse_loss = torch.nn.functional.mse_loss(pred, y)
        self.log(
            "train/loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return mse_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.net(X)
        mse_loss = torch.nn.functional.mse_loss(pred, y)
        self.log(
            "val/loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return mse_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self.net(X)
        mse_loss = torch.nn.functional.mse_loss(pred, y)
        self.log(
            "test/loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        MAE = torch.nn.functional.l1_loss(y, pred)
        self.log(
            "test/MAE",
            MAE,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        MAPE = tm.MeanAbsolutePercentageError().to(self.device)(y, pred)
        self.log(
            "test/MAPE",
            MAPE,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return mse_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        y = self.net(batch)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4
        )
        ret_dict = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train/loss",
        }
        return ret_dict


model = EqNet(
    5,
    200,
    200,
    200,
    2,
)

model.net.load_state_dict(torch.load(net_file))
for name, param in model.named_parameters():
    print(f"name:     {name}")
    print(f"shape:    {param.shape}")
    print(f"req_grad: {param.requires_grad}")
    print("data:")
    print(param.data)

Trainer = pl.Trainer(
    accelerator="cuda",
    enable_progress_bar=False,
)

model.eval()


# dl = torch.utils.data.DataLoader(X, batch_size=100_000, shuffle=False, num_workers=8)
def calc_data():
    # Aufruf der GGW-Funktion und Berechnung der Stoffmengen im GGW
    # print(X_.shape)
    Trainer.predict(model, X_)


def main():
    ns = np.logspace(1, 6, 6, dtype=int, base=10, endpoint=True)
    n_t = np.sum(ns)
    ts = np.empty_like(ns, dtype=float)
    print(ns)
    global X_
    for i, n in enumerate(ns):
        X_ = torch.utils.data.DataLoader(
            X[:n],
            batch_size=int(1e6),
            shuffle=False,
        )

        calc_data()
        t = timeit.timeit(
            "calc_data()",
            globals=globals(),
            number=10,
        )
        print(f"{n:<10}:{t:.5f}")
        v = t / n
        n_t = n_t - n
        print(f"\tit/s: {1/v:.3f}")
        eta = 10 * n_t * v
        print(f"\tETA: {eta:.2f}s")
        ts[i] = t

    np.savez("HSA_001_XXX/timing/time_NN_gpu_1E6.npz", ns=ns, ts=ts)


if __name__ == "__main__":
    main()
