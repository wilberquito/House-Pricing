from dotenv import dotenv_values
import os

envs = ["secret.env", "fit.env"]

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


import torch

torch.cuda.is_available()


# In[5]:


import torch

torch.cuda.device_count()


# In[6]:


import torch

torch.cuda.get_device_name()


# In[7]:


import os

max(1, os.cpu_count() - 1)


# In[8]:


import os
import wandb
import torch
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.model import NeuralNetwork
from src.data import HousePricingDataModule
from src.utils.utility import fake_name

project_name = os.environ["WANDB_NAME"]
iters = int(os.environ["ITERATIONS"])

patience = int(os.environ["PATIENCE"])
max_epochs = int(os.environ["MAX_EPOCHS"])
batch_size = int(os.environ["BATCH_SIZE"])
learning_rate = float(os.environ["LEARNING_RATE"])
validation_size = float(os.environ["VALIDATION_SIZE"])
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Preparing data to be used
data_module = HousePricingDataModule(
    batch_size=batch_size,
    validation_size=validation_size,
)
data_module.prepare_data()

in_features = data_module.data_features()

# Setting up the training configuration
config = {
    "accelerator": accelerator,
    "max_epochs": max_epochs,
    "patience": patience,
    "lr": learning_rate,
    "batch_size": batch_size,
    "in_features": in_features,
    "validation_size": validation_size,
}

artifacts = list()

for i in range(iters):

    run_name = fake_name()

    wandb.init(
        job_type="train",
        name=run_name,
        project=project_name,
        config=config,
    )

    print(f"[INFO]: Fit config: {config}")

    # Defining the model to be training
    model = NeuralNetwork(input_size=wandb.config["in_features"], lr=wandb.config["lr"])

    # Defining the logger instance the lighning will use as default logging
    logger = WandbLogger()

    # Define how the model registry work
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="house_pricing-{epoch:02d}-{val_loss:.2f}",
    )

    # Defining early stop configuration
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=False,
        patience=wandb.config["patience"],
    )

    # Defines the training instance
    trainer = Trainer(
        accelerator=wandb.config["accelerator"],
        max_epochs=wandb.config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, datamodule=data_module)

    artifacts_item = {
        "run_name": run_name,
        "best_model": checkpoint_callback.best_model_path,
        "fit_config": config,
    }
    artifacts.append(artifacts_item)

    wandb.finish()


# In[9]:


tmp = list()
for data in artifacts:
    run_name = data["run_name"]
    best_model = data["best_model"]
    fit_config = data["fit_config"]
    fit_config = {f"fit_{key}": value for key, value in fit_config.items()}
    tmp.append({"run_name": run_name, "best_model": best_model, **fit_config})


# In[10]:


import pandas as pd

pd.DataFrame(tmp)


# In[11]:


import pandas as pd

pd.DataFrame(tmp).to_csv("artifacts.csv", index=None)
