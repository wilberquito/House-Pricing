import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(L.LightningModule):
    def __init__(self, in_features: int):
        super().__init__()

        print(f"[INFO]: Input size: {in_features}")

        self.net = nn.Sequential(
           nn.Linear(in_features=in_features, out_features=512),
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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch["inputs"], batch["target"]
        if target.dim() == 1:
            target = target.view(target.size(0), -1)
        output = self(inputs)
        loss = F.mse_loss(output, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)