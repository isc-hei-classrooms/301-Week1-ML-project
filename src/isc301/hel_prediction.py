import numpy as np
import pandas as pd


def data_preprocessing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def model_fit(X: np.ndarray, Y: np.ndarray) -> None:
    raise NotImplementedError


def model_predict(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError
