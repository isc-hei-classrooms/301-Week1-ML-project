from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.polynomial.polynomial import polyval


def correlation_grid(
    data: pd.DataFrame,
    annot: bool = True,
    triangle: bool = True,
    figsize: tuple[int, int] = (10, 10),
):
    fig, ax = plt.subplots(figsize=figsize)
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) if triangle else np.ones_like(corr)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=annot,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )


Model = Callable[[float], float]


def model_linear(a: float, b: float = 0) -> Model:
    def model(x: float) -> float:
        return a * x + b

    return model


def model_poly(*coeffs: float) -> Model:
    def model(x: float) -> float:
        return np.float64(polyval(x, coeffs))

    return model


def evaluate_mse(model: Model, data_x: np.ndarray, data_y: np.ndarray) -> float:
    y: np.ndarray = model(data_x)  # type: ignore
    mse: float = ((y - data_y) ** 2).mean()
    return mse


def lsm_affine(data_x: np.ndarray, data_y: np.ndarray) -> float:
    denom: float = (data_x * data_y).sum()
    num: float = (data_x * data_x).sum()
    return denom / num


def lsm_linear(data_x: np.ndarray, data_y: np.ndarray) -> tuple[float, float]:
    mean_x: float = data_x.mean()
    mean_y: float = data_y.mean()
    denom: float = (data_x * data_y).mean() - mean_x * mean_y
    num: float = (data_x**2).mean() - mean_x * mean_x
    a: float = denom / num
    b: float = mean_y - a * mean_x
    return a, b
