import os
import subprocess
from sklearn.compose import ColumnTransformer, make_column_selector
import pandas as pd
from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def download_data(script_name: str) -> None:
    if os.path.exists("./data/"):
        print("[INFO]: Skipping downloading data. Data is already downloaded")
    else:
        subprocess.call(["sh", script_name])


def prepare_hold_out_scheme(
    csv_file: str,
    column_transformer: ColumnTransformer,
    test: bool = False,
    test_size: float = 0.1,
    target_column = "SalePrice"
) -> Dict[str, pd.DataFrame]:

    assert test_size >= 0 and test_size <= 1

    df = pd.read_csv(csv_file)

    assert target_column in df.columns

    if test:
        df_train, df_test = \
            train_test_split(df, train_size=1-test_size, random_state=42)

        df_train = preprocess_training_dataframe(df_train)
        df_test = preprocess_dataframe(df_train)

        X_train = df_train.drop(target_column, axis=1)
        X_test = df_test.drop(target_column, axis=1)

        y_train = df_train[target_column]
        y_test = df_test[target_column]

        X_train_preprocessed = column_transformer.fit_transform(X_train)
        X_test_preprocessed = column_transformer.transform(X_test)

        df_train = pd.DataFrame(
            data=X_train_preprocessed,
            columns=column_transformer.get_feature_names_out()
        )
        df_test = pd.DataFrame(
            data=X_test_preprocessed,
            columns=column_transformer.get_feature_names_out()
        )
        df_train[target_column] = y_train
        df_test[target_column] = y_test

        dataframes = {
            "train": df_train,
            "test": df_test
        }
    else:
        X_train = df.drop(target_column, axis=1)
        y_train = df_train[target_column]
        X_train_preprocessed = column_transformer.fit_transform(X_train)

        df_train = pd.DataFrame(
            data=X_train_preprocessed,
            columns=column_transformer.get_feature_names_out()
        )
        df_train[target_column] = y_train

        dataframes = {
            "train": df_train,
        }

    return dataframes


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


def column_transformer() -> ColumnTransformer:

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
