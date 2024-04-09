
import torch
from torch.utils.data import Dataset
import pandas as pd
import lightning as L
from torch.utils.data import random_split, DataLoader

class HousePricingDataset(Dataset):
    """House Pricing dataset."""

    def __init__(self, csv_file, predict=False, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            predict (bool): either to train or to make predictions
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataset_df = pd.read_csv(csv_file)
        dataset_df = dataset_df.drop("Id", axis=1, errors="ignore")
        df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])

        self.predict = predict
        self.transform = transform
        self.features = df_num.drop("SalePrice", axis=1, errors="ignore")
        if not self.predict:
            self.labels = df_num["SalePrice"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.features.iloc[idx]
        X = X.to_numpy()

        if not self.predict:
            y = self.labels.iloc[idx]

        sample = {"features": X, "label": y} if not self.predict else {"features": X}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HousePriceDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.housing_train = None
        self.housing_val = None
        self.housing_test = None
        self.housing_predict = None

    def setup(self, stage: str):
        if stage == "fit":
            housing_full = HousePricingDataset(self.data_dir + "train.csv")
            self.housing_train, self.housing_val, _ = random_split(
                housing_full, [0.75, 0.1, 0.15], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            housing_full = HousePricingDataset(self.data_dir + "train.csv")
            _, _, self.housing_test = random_split(
                housing_full, [0.75, 0.1, 0.15], generator=torch.Generator().manual_seed(42)
            )

        if stage == "predict":
            self.housing_predict = HousePricingDataset(self.data_dir + "predict.csv", predict=True)

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
