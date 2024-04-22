from dotenv import dotenv_values
import os

envs = ["secret.env", "predict.env"]

for fenv in envs:
    file = os.path.join("env", fenv)
    config = dotenv_values(file)  # load sensitive variables
    print(config.keys())
    for c, v in config.items():
        os.environ[c] = v


# In[3]:


import pandas as pd

artifacts = pd.read_csv("artifacts.csv")
artifacts


# In[4]:


artifacts = artifacts.to_dict(orient="records")
artifacts


# In[5]:


from src.data import HousePricingDataModule

data_module = HousePricingDataModule()
data_module.prepare_data()


# In[6]:


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

predictions = dict()

for artifact in artifacts:

    run_name = artifact["run_name"]
    best_model = artifact["best_model"]

    model = NeuralNetwork.load_from_checkpoint(
        checkpoint_path=best_model, input_size=in_features
    )

    trainer = Trainer(accelerator=config["accelerator"])

    prediction = trainer.predict(model, datamodule=data_module)

    predictions[run_name] = prediction


# In[7]:


def prediction_to_submit(prediction_batches):

    submit = {"Id": [], "SalePrice": []}

    for prediction_batch in prediction_batches:
        ids = prediction_batch["id"].tolist()
        predictions = prediction_batch["prediction"].tolist()

        submit["Id"] = submit["Id"] + ids
        submit["SalePrice"] = submit["SalePrice"] + predictions

    return submit


# In[8]:


predictions


# In[10]:


for run_name, prediction in predictions.items():
    submit = prediction_to_submit(prediction)
    submit_csv = pd.DataFrame(submit)
    submit_csv.to_csv(f"submit-{run_name}.csv", index=False)
