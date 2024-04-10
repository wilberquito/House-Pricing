import os
from os.path import join
from typing import Optional
import zipfile
import torch
from torch.utils.data import Dataset
import pandas as pd
import lightning as L
from torch.utils.data import random_split, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from pathlib import Path

def columns_transformer():

    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ("category", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False)),
        ]
    )

    ct = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            make_column_selector(dtype_include=["float64", "int64"])
        ),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=["category"])
        ),
    ])

    return ct


class HousePricingDataset(Dataset):
    """House Pricing dataset."""

    def __init__(self, csv_file: str, predict: bool = False, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            predict (bool): either to train or to make predictions
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset_df = pd.read_csv(csv_file)

        self.predict = predict
        self.transform = transform

        self.samples = dataset_df.drop("SalePrice", axis=1, errors="ignore")
        if not self.predict:
            self.labels = dataset_df["SalePrice"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.samples.iloc[idx]
        X = X.to_numpy()

        if not self.predict:
            y = self.labels.iloc[idx]

        sample = {"sample": X, "label": y} if not self.predict else {"sample": X}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HousePriceDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Optional[str] = None, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir if data_dir else os.getcwd()
        self.data_zip = join(self.data_dir, "data.zip")
        self.data_raw = join(self.data_dir, "raw")
        self.data_preprocessed = join(self.data_dir, "preprocessed")
        Path(self.data_preprocessed).mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size

        self.housing_train = None
        self.housing_val = None
        self.housing_test = None
        self.housing_predict = None

        self.preprocessing = None

    def prepare_data(self) -> None:

        with zipfile.ZipFile(self.data_zip,"r") as zip_ref:
            zip_ref.extractall(self.data_raw)

        train_csv = join(self.data_raw, "all.csv")
        train_df = pd.read_csv(train_csv)
        train_df = self._preprocess_dataframe(train_df)

        X = train_df.drop("SalePrice", axis=1)
        y = train_df[["SalePrice"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        predict_csv = join(self.data_raw, "predict.csv")
        predict_df = pd.read_csv(predict_csv)
        predict_df = self._preprocess_dataframe(predict_df)

        self.preprocessing = preprocessing(X_train)

        c_features = {
            "train": X_train,
            "test": X_test,
            "predict": predict_df
        }

        for stage, X in c_features.items():
            csv_name = join(self.data_preprocessed, stage + ".csv")

            if stage == "train":
                data = self.preprocessing.fit_transform(X)
            else:
                data = self.preprocessing.transform(X)

            df_X = pd.DataFrame(data=data, columns=self.preprocessing.get_feature_names_out())

            df_preprocessed = df_X.copy()

            if stage == "train":
                df_preprocessed["SalePrice"] = y_train["SalePrice"].values
            elif stage == "test":
                transformed_df["SalePrice"] = y_test["SalePrice"].values

            transformed_df.to_csv(csv_name, index=False)

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset_df = df.copy()

        dropped_colums = [
            "Id",
            "Alley",
            "MasVnrType",
            "FireplaceQu",
            "PoolQC",
            "Fence",
            "MiscFeature"
        ]

        dataset_df = dataset_df.drop(dropped_colums, axis=1,
                                    errors="ignore")

        cat_columns = dataset_df.select_dtypes(include = ['O']).columns

        for c in cat_columns:
            dataset_df[c] = dataset_df[c].astype("category")

        return dataset_df

    def setup(self, stage: str):
        if stage == "fit":
            housing_full = HousePricingDataset(join(self.data_preprocessed, "train.csv"))
            self.housing_train, self.housing_val = random_split(
                housing_full, [0.85, 0.15], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            self.housing_test = HousePricingDataset(join(self.data_preprocessed, "test.csv"))

        if stage == "predict":
            self.housing_predict = HousePricingDataset(join(self.data_preprocessed, "predict.csv"), predict=True)

    def train_dataloader(self):
        if not self.housing_train:
            raise Exception("[ERROR]: fit stage not set up")
        return DataLoader(self.housing_train, batch_size=self.batch_size)

    def val_dataloader(self):
        if not self.housing_val:
            raise Exception("[ERROR]: fit stage not set up")
        return DataLoader(self.housing_val, batch_size=self.batch_size)

    def test_dataloader(self):
        if not self.housing_test:
            raise Exception("[ERROR]: test stage not set up")
        return DataLoader(self.housing_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        if not self.housing_predict:
            raise Exception("[ERROR]: predict stage not set up")
        return DataLoader(self.housing_predict, batch_size=self.batch_size)

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

