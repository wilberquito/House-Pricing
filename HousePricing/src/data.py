import os
from os.path import join
from typing import Optional, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
import lightning as L
from torch.utils.data import random_split, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from pathlib import Path

from src.utils.data import download_data, prepare_hold_out_scheme, column_transformer
from src.transforms import ToTensor


class HousePricingDataset(Dataset):
    """House Pricing dataset."""

    def __init__(self, df: pd.DataFrame, predict: bool = False, transform=None):
        """
        Arguments:
            df (pd.DataFrame): dataframe to use as dataset.
            predict (bool): either to train or to make predictions
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.predict = predict
        self.transform = transform

        self.samples = df.drop("SalePrice", axis=1, errors="ignore")
        if not self.predict:
            self.labels = df["SalePrice"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.samples.iloc[idx]
        X = X.to_numpy()

        if not self.predict:
            y = self.labels.iloc[idx]

        sample = {"inputs": X, "target": y} if not self.predict else {"inputs": X}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HousePricingDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        eval_batch_size = 128,
        validate = True,
        test = True,
        validation_size = 0.1,
        test_size = 0.1
    ):

        super().__init__()

        self.validate = validate
        self.test = test

        self.validation_size = validation_size
        self.test_size = test_size

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.data_dir = data_dir if data_dir else join("data")

        self.train_file = join(self.data_dir, "train.csv")
        self.predict_file = join(self.data_dir, "predict.csv")

        self.train_df = None
        self.test_df = None

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.column_transformer = column_transformer()

    def prepare_data(self) -> None:
        download_data()
        self._setup_dataframes()

    def _setup_dataframes(self):
        dataframes = prepare_hold_out_scheme(
            self.train_file,
            *self.column_transformer,
            test=self.test,
            test_size=self.test_size
        )

        self.train_df = dataframes["train"]
        if "test" in dataframes:
            self.test_df = dataframes["test"]

    def setup(self, stage: str):
        print(f"[INFO]: Setting up {stage} dataset/s")
        if stage == "fit":
            self._setup_fit_datasets()
        if stage == "test":
            self._setup_test_dataset()
        if stage == "predict":
            self.predict_dataset = HousePricingDataset(
                csv_file=join(self.data_preprocessed, "predict.csv"), predict=True, transform=ToTensor()
            )

    def _setup_fit_datasets(self) -> None:
        train_dataset = HousePricingDataset(
            self.train_df,
            transform=ToTensor()
        )
        if self.validate:
            self.train_dataset, self.validation_dataset = random_split(
                train_dataset,
                [1 - self.validation_size, self.validation_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"[INFO]: Train dataset size: {len(self.train_dataset)}")
            print(f"[INFO]: Validation dataset size: {len(self.validation_dataset)}")
        else:
            self.train_dataset = train_dataset
            print(f"[INFO]: Train dataset size: {len(self.train_dataset)}")

    def _setup_test_dataset(self) -> None:
        self.test_dataset = HousePricingDataset(
            self.test_df,
            transform=ToTensor()
        )

    def train_dataloader(self):
        if not self.train_dataset:
            raise Exception("[ERROR]: fit stage not set up")
        # return DataLoader(self.housing_train, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        print(f"[INFO]: Train dataloader size: {len(dl)}")
        return dl

    def val_dataloader(self):
        if not self.validation_dataset:
            raise Exception("[ERROR]: fit stage not set up")
        # return DataLoader(self.housing_val, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.validation_dataset, batch_size=self.eval_batch_size, shuffle=True)
        print(f"[INFO]: Validation dataloader size: {len(dl)}")
        return dl

    def test_dataloader(self):
        if not self.test_dataset:
            raise Exception("[ERROR]: test stage not set up")
        # return DataLoader(self.housing_test, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.test_dataset, batch_size=self.eval_batch_size)
        print(f"[INFO]: Test dataloader size: {len(dl)}")
        return dl

    def predict_dataloader(self):
        if not self.predict_dataset:
            raise Exception("[ERROR]: predict stage not set up")
        # return DataLoader(self.housing_predict, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.predict_dataset, batch_size=self.eval_batch_size)
        print(f"[INFO]: Predict dataloader size: {len(dl)}")
        return dl

    def in_features(self) -> int:
        """Computes the number of features expected for the dataset
        """
        train_csv = join(self.data_preprocessed, "train" + ".csv")
        if not os.path.isfile(train_csv):
            print("[WARN]: preprocessed dataset are not created... setting up them")
            self.setup()

        df = pd.read_csv(train_csv)
        _, in_features = df.shape
        return in_features - 1


