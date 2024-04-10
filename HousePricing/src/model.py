import lightning as L
import torch
import torch.nn as nn


class HousePricingModel(L.LightningModule):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
           nn.Linear(in_features=in_features, out_features=in_features*3),
           nn.ReLU(),
           nn.Linear(in_features=in_features*3, out_features=in_features*2),
           nn.ReLU(),
           nn.Linear(in_features=in_features*2, out_features=min(in_features, 10)),
           nn.ReLU(),
           nn.Linear(in_features=min(in_features, 10), out_features=1),
        )

    def forward(self, inputs):
        return self.net(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch["inputs"], batch["target"]
        output = self.net(inputs)
        loss = torch.nn.MSELoss(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch["inputs"], batch["target"]
        output = self.net(inputs)
        loss = torch.nn.MSELoss(output, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=0.1)