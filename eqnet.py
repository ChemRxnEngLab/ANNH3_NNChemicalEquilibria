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

wandb.login()

torch.set_default_dtype(torch.float64)


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
        self.linear_afunc_stack = nn.Sequential(
            # nn.BatchNorm1d(input_size), # Normalisierung, damit Inputdaten gleiche Größenordnung haben
            nn.Linear(
                input_size, hidden1_size
            ),  # Lineare Transformation mit gespeicherten weights und biases
            nn.GELU(),  # Nicht lineare Aktivierungsfunktion um komplexe nichtlineare Zusammenhänge abzubilden
            nn.Linear(hidden1_size, hidden2_size),
            nn.GELU(),
            nn.Linear(hidden2_size, hidden3_size),
            nn.GELU(),
            nn.Linear(hidden3_size, output_size),
        )

    # Implementierung der Operationen auf Input Daten
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear_afunc_stack(x)
        return out


class DataModule(pl.LightningDataModule):
    def __init__(self, path_1, path_2, batch_size=256):
        super().__init__()
        self.path_1 = path_1
        self.path_2 = path_2
        self.batch_size = batch_size

    def prepare_data(self):
        data_1 = np.load(self.path_1)
        data_2 = np.load(self.path_2)
        T = torch.cat(
            (torch.tensor(data_1["T"])[:500], torch.tensor(data_2["T"])[:500]), 0
        )
        p = torch.cat(
            (torch.tensor(data_1["p"])[:500], torch.tensor(data_2["p"])[:500]), 0
        )
        x_0 = torch.cat(
            (torch.tensor(data_1["x_0"][:500]), torch.tensor(data_2["x_0"])[:500]), 0
        )
        x_eq = torch.cat(
            (torch.tensor(data_1["x"][:500]), torch.tensor(data_2["x"])[:500]), 0
        )

        # print(T.shape, p.shape, x_0.shape, x_eq.shape)

        X = torch.cat((T[:, None], p[:, None], x_0), 1)
        y = x_eq[:, [0, 2]]

        self.X_mean = X.mean(0)
        self.X_std = X.std(0)
        self.y_mean = y.mean(0)
        self.y_std = y.std(0)

        X_norm = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True)
        y_norm = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True)
        # print(X_norm)

        self.data = TensorDataset(X_norm, y_norm)
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

        y_true = y * self.trainer.datamodule.y_std.to(
            self.device
        ) + self.trainer.datamodule.y_mean.to(self.device)
        y_pred = pred * self.trainer.datamodule.y_std.to(
            self.device
        ) + self.trainer.datamodule.y_mean.to(self.device)

        MAE = torch.nn.functional.l1_loss(y_true, y_pred)
        self.log(
            "test/MAE",
            MAE,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        MAPE = tm.MeanAbsolutePercentageError().to(self.device)(y_true, y_pred)
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
        y = (
            self.net(batch) * self.trainer.datamodule.y_std
            + self.trainer.datamodule.y_mean
        )
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


model = EqNet(5, 200, 200, 200, 2)
dm = DataModule(
    path_1=Path.cwd() / "data" / "eq_dataset_x.npz",
    path_2=Path.cwd() / "data" / "eq_dataset_x_extra.npz",
    batch_size=32,
)

dm.prepare_data()

my_logger = pl.loggers.WandbLogger(
    project="eqnet",
    save_dir=Path.cwd() / "logs",
    log_model=True,
    tags=["eqnet", "pytorch"],
)

Trainer = pl.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=1000,
    # callbacks=[
    #     lr_callback,
    #     test_logging_callback,
    # ],
    logger=my_logger,
    # deterministic=True,
    # enable_progress_bar=False,
)

Trainer.fit(model, dm)
Trainer.test(model, dm)

torch.save(
    model.net.state_dict,
    Path.cwd() / "models" / "torch" / "NH3_net.pt",
)
model.to_onnx(
    Path.cwd() / "models" / "onnx" / "NH3_net.onnx",
    input_sample=torch.rand((5,)),
    export_params=True,
)

wandb.finish()
