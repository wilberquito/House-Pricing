from dotenv import dotenv_values
import os

envs = ["secret.env", "test.env"]

for fenv in envs:
    file = os.path.join("env", fenv)
    config = dotenv_values(file)  # load sensitive variables
    print(config.keys())
    for c, v in config.items():
        os.environ[c] = v


# In[3]:


import wandb
import os

wandb_key = os.environ["WANDB_API_KEY"]
wandb.login(key=wandb_key)


# In[4]:


# Preparing data to be used
from src.data import HousePricingDataModule

data_module = HousePricingDataModule()
data_module.prepare_data()


# In[5]:


import pandas as pd

artifacts = pd.read_csv("artifacts.csv")
artifacts


# In[6]:


artifacts = artifacts.to_dict(orient="records")
artifacts


# In[7]:


import os
from lightning import Trainer
import torch

from pytorch_lightning.loggers import WandbLogger

from src.model import NeuralNetwork

project_name = os.environ["WANDB_NAME"]

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
in_features = data_module.data_features()

config = {
    "accelerator": accelerator,
    "in_features": in_features,
}

test_results = list()

for artifact in artifacts:

    run_name = artifact["run_name"]
    best_model = artifact["best_model"]

    wandb.init(
        job_type="test",
        name=run_name,
        project=project_name,
        config=config,
    )

    model = NeuralNetwork.load_from_checkpoint(
        checkpoint_path=best_model, input_size=in_features
    )

    logger = WandbLogger(checkpoint_name=run_name)

    trainer = Trainer(
        accelerator=wandb.config["accelerator"], logger=logger, log_every_n_steps=1
    )

    test_loss = trainer.test(model, datamodule=data_module)

    data = {"run_name": run_name, "best_model": best_model, "test_loss": test_loss}

    test_results.append(data)

    wandb.finish()


# In[8]:


test_results
