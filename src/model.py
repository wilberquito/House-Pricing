from os.path import join
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import RMSELoss


class NeuralNetwork(L.LightningModule):
    def __init__(self, input_size: int, lr=0.001):
        super().__init__()

        self.input_size = input_size
        self.lr = lr

        self.loss_fn = F.mse_loss

        self.net = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, inputs):
        output = self.net(inputs)
        return output

    def training_step(self, batch):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss = self._share_eval_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        loss = self._share_eval_step(batch)
        self.log("test_loss", loss)
        return loss

    def _share_eval_step(self, batch):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = self.loss_fn(output, target)
        return loss

    def predict_step(self, batch):
        ids = batch["id"]
        inputs = batch["inputs"]
        output = self(inputs).flatten()
        return {"id": ids, "prediction": output}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_start(self) -> None:
        logger = self.trainer.logger
        self.save_model_dir = join(logger.save_dir, logger.name, logger.version)
        print(f"[INFO]: Logger save model dir: {self.save_model_dir}")


class HousePricingNN(NeuralNetwork):
    def __init__(self, input_size: int, lr=0.001):
        super().__init__(input_size, lr)

        # self.loss_fn = RMSELoss()
        self.loss_fn = F.mse_loss

        self.net = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=32, out_features=1),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
