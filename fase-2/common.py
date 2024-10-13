import pandas as pd


def fill_nan_with_mode(column):
    mode = column.mode()[0]
    return column.fillna(mode)


def fill_value_with_mode(column, value: str):
    mode = column.mode()[0]
    return column.replace(value, mode)


def preprocess_data(X: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    X["workclass"] = fill_nan_with_mode(X["workclass"])
    X["native-country"] = fill_nan_with_mode(X["native-country"])

    X["workclass"] = fill_value_with_mode(X["workclass"], "?")
    X["native-country"] = fill_value_with_mode(X["native-country"], "?")

    X["occupation"] = X["occupation"].fillna("?")

    X.drop(columns=["education"], inplace=True)

    X = pd.get_dummies(
        X,
        columns=[
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
    )

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)

    return X


def preprocess_output(y: pd.DataFrame) -> pd.DataFrame:
    y[y == ">50K."] = ">50K"
    y[y == "<=50K."] = "<=50K"
    y[y == ">50K"] = 1
    y[y == "<=50K"] = 0

    return y.astype(int)
