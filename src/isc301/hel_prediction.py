import abc
from typing import Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from isc301.utils import lsm_linear


class Model(abc.ABC):
    def __init__(self) -> None:
        self.final_df: Optional[pd.DataFrame] = None

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pass

    def split_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        split1: list[np.ndarray] = train_test_split(X, y)  # type: ignore
        X_train, X2, y_train, y2 = split1
        print(X_train.shape, X2.shape, y_train.shape, y2.shape)
        split2: list[np.ndarray] = train_test_split(X2, y2)  # type: ignore
        X_validate, X_test, y_validate, y_test = split2
        return (X_train, y_train), (X_validate, y_validate), (X_test, y_test)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_real: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], float],
    ) -> float:
        y_test = self.predict(X_test)
        return metric(y_real, y_test)

    def process_dataset(
        self, df: pd.DataFrame, metric: Callable[[np.ndarray, np.ndarray], float]
    ) -> pd.DataFrame:
        X, y = self.preprocess(df)
        train, validate, test = self.split_dataset(X, y)
        self.fit(*train)
        predicted: np.ndarray = self.predict(validate[0])
        residuals: np.ndarray = predicted - validate[1]
        train_score: float = self.evaluate(train[0], train[1], metric)
        validate_score: float = self.evaluate(validate[0], validate[1], metric)

        all_predicted: np.ndarray = self.predict(X)
        residuals: np.ndarray = all_predicted - y
        df2 = df.assign(predicted=all_predicted, residual=residuals)
        print(f"Train Score: {train_score}")
        print(f"Validate Score: {validate_score}")
        self.final_df = df2
        return df2

    def explain(self):
        pass

    @property
    def vars(self):
        return []

    def plot_residuals(self):
        if self.final_df is None:
            return
        fig, axs = plt.subplots(
            1,
            len(self.vars),
            figsize=(len(self.vars) * 3, 4),
            sharey="all",
        )
        for i, var in enumerate(self.vars):
            residuals = self.final_df["residual"]
            print(f"{var=} min={residuals.min()} max={residuals.max()}")
            axs[i].scatter(self.final_df[var], residuals)
            axs[i].set_xlabel(var)
            axs[i].set_ylabel("Residual")


class LinearModel(Model):
    def __init__(self, x_col: str, y_col: str) -> None:
        super().__init__()
        self.x_col: str = x_col
        self.y_col: str = y_col
        self.a: float = 0
        self.b: float = 0

    @property
    def vars(self):
        return [self.x_col]

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.array(df[self.x_col]), np.array(df[self.y_col])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.a, self.b = lsm_linear(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.a * X + self.b

    def explain(self):
        print("Linear model:")
        print(f"- a={self.a}")
        print(f"- b={self.b}")


class PolyModel(Model):
    def __init__(
        self, x_cols: str | list[str], y_col: str, reg, degree: int = 2
    ) -> None:
        super().__init__()
        self.x_cols: list[str] = [x_cols] if isinstance(x_cols, str) else x_cols
        self.y_col: str = y_col
        self.degree: int = degree
        self.features: PolynomialFeatures = PolynomialFeatures(self.degree)
        self.model = reg
        # self.scaler: StandardScaler = StandardScaler()
        self.scaler: MinMaxScaler = MinMaxScaler()

    @property
    def vars(self):
        return self.x_cols

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.array(df[self.x_cols]), np.array(df[self.y_col])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        X_poly = self.features.fit_transform(X_scaled)
        self.model.fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_poly = self.features.fit_transform(X_scaled)
        return self.model.predict(X_poly)

    def explain(self):
        print("Polynomial model:")
        print(self.model.coef_)


def data_preprocessing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def model_fit(X: np.ndarray, Y: np.ndarray) -> None:
    raise NotImplementedError


def model_predict(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError
