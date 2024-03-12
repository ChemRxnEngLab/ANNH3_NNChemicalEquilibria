from typing import Any
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics as tm
import numpy as np
from pathlib import Path
import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

import sys

sys.path.append(str(Path.cwd() / "lib_nets"))
from Nets.EQ_Net_A import NeuralNetwork

wandb.login()

torch.set_default_dtype(torch.float64)


class DataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=256):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def prepare_data(self):
        data_1 = np.load(self.path)
        T = torch.tensor(data_1["T"])
        p = torch.tensor(data_1["p"])
        x_0 = torch.tensor(data_1["x_0"])
        x_eq = torch.tensor(data_1["x"])

        # print(T.shape, p.shape, x_0.shape, x_eq.shape)

        X = torch.cat((T[:, None], p[:, None], x_0), 1)
        y = x_eq[:, [0, 2]]

        self.X_mean = X.mean(0)
        self.X_std = X.std(0)
        self.y_mean = y.mean(0)
        self.y_std = y.std(0)

        self.data = TensorDataset(X, y)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.data, [0.7, 0.1, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


class EqNet(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        hidden3_size: int,
        output_size: int,
        mean_in: torch.Tensor,
        mean_out: torch.Tensor,
        std_in: torch.Tensor,
        std_out: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = NeuralNetwork(
            input_size,
            hidden1_size,
            hidden2_size,
            hidden3_size,
            output_size,
            mean_in,
            mean_out,
            std_in,
            std_out,
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


dm_uniform = DataModule(
    path=Path.cwd() / "HSA_001_XXX" / "data" / "HSA_003_eq_dataset_uniform.npz",
    batch_size=256,
)

dm_uniform.prepare_data()

dm_loguniform = DataModule(
    path=Path.cwd() / "HSA_001_XXX" / "data" / "HSA_003_eq_dataset_loguniform.npz",
    batch_size=256,
)

dm_loguniform.prepare_data()

dm = dm_loguniform

model = EqNet(
    5,
    200,
    200,
    200,
    2,
    mean_in=dm.X_mean,
    mean_out=dm.y_mean,
    std_in=dm.X_std,
    std_out=dm.y_std,
)

my_logger = WandbLogger(
    project="NH3_eqnet",
    save_dir=Path.cwd() / "logs",
    log_model=True,
    tags=["eqnet"],
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

Trainer = pl.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=1000,
    callbacks=[
        lr_monitor,
    ],
    logger=my_logger,
    deterministic=True,
)

Trainer.fit(model, dm)
print("Uniform Net on loguniform dataloader")
Trainer.test(model, dm_loguniform)
print("Uniform net on uniform dataloader")
Trainer.test(model, dm_uniform)
model.eval()

wandb.finish()

torch.save(
    model.net.state_dict(),
    Path.cwd() / "models" / "torch" / "NH3_net_SL.pt",
)

# onnx_program = torch.onnx.dynamo_export(model.net, torch.rand((5,)))
# onnx_program.save(Path.cwd() / "models" / "onnx" / "extended_NH3_net.onnx")
