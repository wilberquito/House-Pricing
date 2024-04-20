import os
import subprocess
from sklearn.compose import ColumnTransformer, make_column_selector
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


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
    train_csv: str,
    column_transformer: ColumnTransformer,
    predict_csv=None,
    test=False,
    predict=False,
    test_size=0.1,
    target="SalePrice",
) -> Dict[str, pd.DataFrame]:

    fit_df = None
    test_df = None
    predict_df = None

    fit_df = pd.read_csv(train_csv)
    dropped_columns = []

    if test:
        fit_df, test_df = train_test_split(
            fit_df, train_size=1 - test_size, random_state=42
        )

        fit_df, dropped_columns = _preprocess_fit_dataframe(fit_df)
        test_df = _preprocess_eval_dataframe(test_df, dropped_columns)

        X_train = column_transformer.fit_transform(fit_df.drop(target, axis=1))
        X_test = column_transformer.transform(test_df.drop(target, axis=1))

        y_train = fit_df[target]
        y_test = test_df[target]

        fit_df = pd.DataFrame(
            data=X_train, columns=column_transformer.get_feature_names_out()
        )
        fit_df[target] = y_train

        test_df = pd.DataFrame(
            data=X_test, columns=column_transformer.get_feature_names_out()
        )
        test_df[target] = y_test
    else:
        fit_df, dropped_columns = _preprocess_fit_dataframe(fit_df)

        X_train = column_transformer.fit_transform(fit_df.drop(target, axis=1))
        y_train = fit_df[target]

        fit_df = pd.DataFrame(
            data=X_train, columns=column_transformer.get_feature_names_out()
        )
        fit_df[target] = y_train

    if predict:
        predict_df = pd.read_csv(predict_csv)
        predict_df = _preprocess_eval_dataframe(predict_df, dropped_columns)

        X_predict = column_transformer.transform(predict_df)

        predict_df = pd.DataFrame(
            data=X_predict, columns=column_transformer.get_feature_names_out()
        )

    return {
        "fit": fit_df,
        "test": test_df,
        "predict": predict_df,
    }


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
        "Id",
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
