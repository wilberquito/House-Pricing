from os.path import join
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F



class NeuralNetwork(L.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()

        self.input_size = input_size

        print(f"[INFO]: Input size: {self.input_size}")

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

    def training_step(self, batch, batch_idx):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch):
        inputs = batch["inputs"]
        output = self(inputs)
        return output

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)

    def on_train_start(self) -> None:
        logger = self.trainer.logger
        self.save_model_dir = join(logger.save_dir, logger.name, logger.version)
        print(f"[INFO]: Logger save model dir: {self.save_model_dir}")

