import os
from os.path import join
from pathlib import Path
import subprocess
from sklearn.compose import ColumnTransformer, make_column_selector
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def submit_prediction(script_name: str) -> None:
    subprocess.call(["sh", script_name])


def download_data(script_name: str) -> None:
    if os.path.exists("./data/"):
        print("[INFO]: Skipping downloading data. Data is already downloaded")
    else:
        subprocess.call(["sh", script_name])


def column_transformer() -> ColumnTransformer:

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("category", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ct = ColumnTransformer(
        [
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include=["float64", "int64"]),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=["category"]),
            ),
        ]
    )

    return ct


def setup_dataframes(
    data_dir: str,
    column_transformer: ColumnTransformer,
    test_size=0.1,
    id_column_name="Id",
    target_column_name="SalePrice",
) -> Dict[str, pd.DataFrame]:

    ready_data_dir = join(data_dir, "ready")

    if os.path.exists(ready_data_dir):
        dataframes = _reload_setup_dataframes(data_dir)
    else:
        dataframes = _fresh_setup_dataframes(
            data_dir, column_transformer, test_size, id_column_name, target_column_name
        )

    return dataframes


def _fresh_setup_dataframes(
    data_dir: str,
    column_transformer: ColumnTransformer,
    test_size: int,
    id_column_name="Id",
    target_column_name="SalePrice",
) -> Dict[str, pd.DataFrame]:

    print("[INFO]: Fresh data setup...")
    ready_data_dir = join(data_dir, "ready")
    Path(ready_data_dir).mkdir(exist_ok=True, parents=True)

    data_fit_test_dataframes = _setup_fit_test_dataframes(
        data_dir,
        column_transformer,
        test_size=test_size,
        id_column_name=id_column_name,
        target_column_name=target_column_name,
    )

    dropped_columns = data_fit_test_dataframes["dropped_columns"]

    fit_test_dataframes = data_fit_test_dataframes["data_frames"]
    predict_dataframe = _setup_predict_dataframes(
        data_dir,
        column_transformer,
        id_column_name=id_column_name,
        dropped_columns=dropped_columns,
    )

    dataframes: Dict[str, pd.DataFrame] = {**fit_test_dataframes, **predict_dataframe}

    for name, df in dataframes.items():
        csv_name = join(ready_data_dir, f"{name}.csv")
        df.to_csv(csv_name, index=False)

    return dataframes


def _reload_setup_dataframes(data_dir: str) -> Dict[str, pd.DataFrame]:
    print("[INFO]: Reloading set up data...")
    csv_names = ["fit", "test", "predict"]
    ready_data_dir = join(data_dir, "ready")
    dataframes = {csv: pd.read_csv(join(ready_data_dir, f"{csv}.csv")) for csv in csv_names}

    return dataframes


def _setup_fit_test_dataframes(
    data_dir: str,
    column_transformer: ColumnTransformer,
    test_size=0.1,
    id_column_name="Id",
    target_column_name="SalePrice",
) -> Dict[str, Any]:

    train_csv = join(data_dir, "train.csv")

    fit_df = pd.read_csv(train_csv)
    test_df = None

    dropped_columns = []

    fit_df, test_df = train_test_split(
        fit_df, train_size=1 - test_size, random_state=42
    )

    fit_df, dropped_columns = _preprocess_fit_dataframe(fit_df)
    test_df = _preprocess_eval_dataframe(test_df, dropped_columns)

    fit_X = column_transformer.fit_transform(
        fit_df.drop([id_column_name, target_column_name], axis=1)
    )
    test_X = column_transformer.transform(
        test_df.drop([id_column_name, target_column_name], axis=1)
    )

    fit_y = fit_df[target_column_name]
    test_y = test_df[target_column_name]

    fit_id = fit_df[id_column_name]
    test_id = test_df[id_column_name]

    fit_df = pd.DataFrame(
        data=fit_X, columns=column_transformer.get_feature_names_out()
    )
    fit_df[target_column_name] = fit_y
    fit_df[id_column_name] = fit_id

    test_df = pd.DataFrame(
        data=test_X, columns=column_transformer.get_feature_names_out()
    )
    test_df[target_column_name] = test_y
    test_df[id_column_name] = test_id

    dataframes = {"fit": fit_df, "test": test_df}

    return {"data_frames": dataframes, "dropped_columns": dropped_columns}


def _setup_predict_dataframes(
    data_dir: str,
    column_transformer: ColumnTransformer,
    id_column_name="Id",
    dropped_columns=[],
) -> Dict[str, pd.DataFrame]:

    predict_csv = join(data_dir, "predict.csv")

    predict_df = pd.read_csv(predict_csv)
    predict_df = _preprocess_eval_dataframe(predict_df, dropped_columns)

    predict_X = column_transformer.transform(predict_df.drop(id_column_name, axis=1))
    predict_id = predict_df[id_column_name]

    predict_df = pd.DataFrame(
        data=predict_X, columns=column_transformer.get_feature_names_out()
    )
    predict_df[id_column_name] = predict_id

    return {"predict": predict_df}


def _preprocess_fit_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dataset_df = df.copy().reset_index(drop=True)
    dataset_df = _preprocess_dataframe(dataset_df)

    cat_columns = dataset_df.select_dtypes(include=["category"]).columns

    print(
        f"[INFO]: Dropping categories types with few ocurrence. Current dataframe shape: {dataset_df.shape}"
    )
    dropping_ocurrences = {}

    for c in cat_columns:
        counts = dataset_df[c].value_counts().to_dict()
        targets = [target for target, value in counts.items() if value <= 1]
        if len(targets) >= 1:
            dropping_ocurrences[c] = targets

    for c, targets in dropping_ocurrences.items():
        dataset_df = dataset_df[~dataset_df[c].isin(targets)].reset_index(drop=True)

    print(
        f"[INFO]: Few ocurrences removed. Current dataframe shape: {dataset_df.shape}"
    )

    print(
        f"[INFO]: Dropping columns which contains just one type of category. Current dataframe shape: {dataset_df.shape}"
    )
    dropped_columns = []

    for c in cat_columns:
        counts = dataset_df[c].value_counts().to_dict()
        counts = {c: count for c, count in counts.items() if count >= 2}
        if len(counts) <= 1:
            dropped_columns.append(c)

    dataset_df = dataset_df.drop(dropped_columns, errors="ignore", axis=1).reset_index(
        drop=True
    )

    print(
        f"[INFO]: Columns with just one type of categroy dropped. Current dataframe shape: {dataset_df.shape}"
    )

    return dataset_df, dropped_columns


def _preprocess_eval_dataframe(df: pd.DataFrame, drop_columns=[]) -> pd.DataFrame:
    df = _preprocess_dataframe(df)
    df = df.drop(columns=drop_columns, axis=1)
    return df


def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    dataset_df = df.copy().reset_index(drop=True)

    print(
        f"[INFO]: Dropping columns with full of NA or Identifiers. Current dataframe shape: {dataset_df.shape}"
    )

    dropped_colums = [
        "Alley",
        "MasVnrType",
        "FireplaceQu",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]

    dataset_df = dataset_df.drop(dropped_colums, axis=1, errors="ignore")

    print(
        f"[INFO]: Dropped columns with full of NA or Identifiers. Current dataframe shape: {dataset_df.shape}"
    )

    cat_columns = dataset_df.select_dtypes(include=["O"]).columns

    for c in cat_columns:
        dataset_df[c] = dataset_df[c].astype("category")

    return dataset_df
