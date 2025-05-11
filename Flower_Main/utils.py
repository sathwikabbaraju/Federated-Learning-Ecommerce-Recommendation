from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Types for clarity
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Extracts coefficients (and intercept) from a scikit-learn LR."""
    if model.fit_intercept:
        return [model.coef_, model.intercept_]
    return [model.coef_]


def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    """Injects coefficients (and intercept) into a scikit-learn LR."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Creates zeroed parameters so Flower can request an init state."""
    # We know it's binary purchased → 2 classes
    n_classes = 2  
    # Infer feature count from client1’s training data
    X_train, _ = load_data_client1()[0]
    n_features = X_train.shape[1]

    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def load_data_client1() -> Dataset:
    """Load and split the ‘impulsivemax’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_impulsivemax.csv"
    )
    # Drop non-numeric columns; assume 'purchased' is your target
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    # Standardize and stratified split
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client2() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_impulsivemid.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client3() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_middiscount.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client4() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_midvalue.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client5() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_ratermax.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client6() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_ratermed.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client7() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_saverfree.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client8() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_savermed.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client9() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_saverpro.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client10() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_socialproof.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def load_data_client11() -> Dataset:
    """Load and split the ‘midvalue’ persona dataset."""
    df = pd.read_csv(
        "/Users/sathwik/VISUAL STUDIO CODE/FLEC/synthetic_dataset/products_valuerater.csv"
    )
    X = df.drop(columns=["name", "main_category", "sub_category", "purchased"]).values
    y = df["purchased"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle features and labels in unison."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split the dataset into `num_partitions` roughly equal chunks."""
    return list(zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))
