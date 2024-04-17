import os
from os.path import join
from typing import Optional
import zipfile
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

from src.util import optim_workers
from src.transforms import ToTensor


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = df.copy().reset_index(drop=True)

    print(f"[INFO]: Dropping columns with full of NA or Identifiers. Current dataframe shape: {dataset_df.shape}")

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

    print(f"[INFO]: Dropped columns with full of NA or Identifiers. Current dataframe shape: {dataset_df.shape}")

    cat_columns = dataset_df.select_dtypes(include = ['O']).columns

    for c in cat_columns:
        dataset_df[c] = dataset_df[c].astype("category")

    return dataset_df


def preprocess_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = df.copy().reset_index(drop=True)
    dataset_df = preprocess_dataframe(dataset_df)

    cat_columns = dataset_df.select_dtypes(include = ["category"]).columns

    print(f"[INFO]: Dropping categories types with few ocurrence. Current dataframe shape: {dataset_df.shape}")
    dropping_ocurrences = {}

    for c in cat_columns:
        counts = dataset_df[c].value_counts().to_dict()
        targets = [target for target, value in counts.items() if value <= 1]
        if len(targets) >= 1:
            dropping_ocurrences[c] = targets

    for c, targets in dropping_ocurrences.items():
        dataset_df = dataset_df[~dataset_df[c].isin(targets)].reset_index(drop=True)

    print(f"[INFO]: Few ocurrences removed. Current dataframe shape: {dataset_df.shape}")

    print(f"[INFO]: Dropping columns which contains just one type of category. Current dataframe shape: {dataset_df.shape}")
    cols_to_drop = []

    for c in cat_columns:
        counts = dataset_df[c].value_counts().to_dict()
        counts = {c: count for c, count in counts.items() if count >= 2}
        if len(counts) <= 1:
            cols_to_drop.append(c)

    dataset_df = dataset_df.drop(cols_to_drop, errors="ignore", axis=1).reset_index(drop=True)

    print(f"[INFO]: Columns with just one type of categroy dropped. Current dataframe shape: {dataset_df.shape}")

    return dataset_df


def columns_transformer():

    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ("category", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

        sample = {"inputs": X, "target": y} if not self.predict else {"inputs": X}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HousePricingDataModule(L.LightningDataModule):
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

        self.columns_transformer = None

    def prepare_data(self) -> None:

        with zipfile.ZipFile(self.data_zip,"r") as zip_ref:
            zip_ref.extractall(self.data_raw)

        train_csv = join(self.data_raw, "all.csv")
        train_df = pd.read_csv(train_csv)

        print("[INFO]: Preprocessing training dataframe...")
        train_df = preprocess_training_dataframe(train_df)

        X = train_df.drop("SalePrice", axis=1)
        y = train_df[["SalePrice"]]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.8, random_state=42)

        print("[INFO]: Preprocessing predict dataframe...")
        predict_csv = join(self.data_raw, "predict.csv")
        predict_df = pd.read_csv(predict_csv)
        predict_df = preprocess_dataframe(predict_df)

        self.columns_transformer = columns_transformer()

        c_features = {
            "train": X_train,
            "test": X_test,
            "predict": predict_df
        }

        for stage, X in c_features.items():
            csv_name = join(self.data_preprocessed, stage + ".csv")

            if stage == "train":
                data = self.columns_transformer.fit_transform(X)
            else:
                data = self.columns_transformer.transform(X)

            df_X = pd.DataFrame(data=data, columns=self.columns_transformer.get_feature_names_out())

            transformed_df = df_X.copy()

            if stage == "train":
                transformed_df["SalePrice"] = y_train["SalePrice"].values
            elif stage == "test":
                transformed_df["SalePrice"] = y_test["SalePrice"].values

            transformed_df.to_csv(csv_name, index=False)

    def setup(self, stage: str):
        print(f"[INFO]: Setting up {stage} dataset/s")
        if stage == "fit":
            housing_full = HousePricingDataset(
                csv_file=join(self.data_preprocessed, "train.csv"), transform=ToTensor()
            )
            self.housing_train, self.housing_val = random_split(
                housing_full, [0.85, 0.15], generator=torch.Generator().manual_seed(42)
            )
            print(f"[INFO]: Train dataset size: {len(self.housing_train)}")
            print(f"[INFO]: Validation dataset size: {len(self.housing_val)}")

        if stage == "test":
            self.housing_test = HousePricingDataset(
                csv_file=join(self.data_preprocessed, "test.csv"), transform=ToTensor()
            )

        if stage == "predict":
            self.housing_predict = HousePricingDataset(
                csv_file=join(self.data_preprocessed, "predict.csv"), predict=True, transform=ToTensor()
            )

    def train_dataloader(self):
        if not self.housing_train:
            raise Exception("[ERROR]: fit stage not set up")
        # return DataLoader(self.housing_train, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.housing_train, batch_size=self.batch_size, shuffle=True)
        print(f"[INFO]: Train dataloader size: {len(dl)}")
        return dl

    def val_dataloader(self):
        if not self.housing_val:
            raise Exception("[ERROR]: fit stage not set up")
        # return DataLoader(self.housing_val, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.housing_val, batch_size=self.batch_size, shuffle=True)
        print(f"[INFO]: Validation dataloader size: {len(dl)}")
        return dl

    def test_dataloader(self):
        if not self.housing_test:
            raise Exception("[ERROR]: test stage not set up")
        # return DataLoader(self.housing_test, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.housing_test, batch_size=self.batch_size)
        print(f"[INFO]: Test dataloader size: {len(dl)}")
        return dl

    def predict_dataloader(self):
        if not self.housing_predict:
            raise Exception("[ERROR]: predict stage not set up")
        # return DataLoader(self.housing_predict, batch_size=self.batch_size, num_workers=optim_workers())
        dl = DataLoader(self.housing_predict, batch_size=self.batch_size)
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


