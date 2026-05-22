import pandas as pd


def load_data(file_paths):
    df_train = pd.read_csv(file_paths["train_data"])
    X_train = df_train[["x1", "x2"]].to_numpy()
    y_train = df_train["y"].to_numpy()

    df_test = pd.read_csv(file_paths["test_data"])
    X_test = df_test[["x1", "x2"]].to_numpy()
    y_test = df_test["y"].to_numpy()
    return X_train, y_train, X_test, y_test
