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
from Nets.EQ_Net_B import NeuralNetwork

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
        norm_momentum: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = NeuralNetwork(
            input_size,
            hidden1_size,
            hidden2_size,
            hidden3_size,
            output_size,
            norm_momentum,
        )
        self.save_hyperparameters()

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
        mean_p = self.net.norm.running_mean[1]
        var_p = self.net.norm.running_var[1]
        self.log(
            "val/loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val/mean_p",
            mean_p,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val/var_p",
            var_p,
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


dm = DataModule(
    path=Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset_uniform.npz",
    batch_size=256,
)

dm.prepare_data()

model = EqNet(
    5,
    200,
    200,
    200,
    2,
    norm_momentum=0.001,
)
my_logger = WandbLogger(
    project="NH3_eqnet",
    save_dir=Path.cwd() / "logs",
    log_model=True,
    tags=["BatchNorm"],
)

# my_logger.watch(model, log="all", log_freq=100, log_graph=False)

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
Trainer.test(model, dm)
model.eval()

wandb.finish()

torch.save(
    model.net.state_dict(),
    Path.cwd() / "models" / "torch" / "BatchNorm_NH3_net_uniform.pt",
)
# model.to_onnx(
#     Path.cwd() / "models" / "onnx" / "extended_NH3_net_BatchNorm.onnx",
#     input_sample=torch.rand((5,)),
#     export_params=True,
# )
