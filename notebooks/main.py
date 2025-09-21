from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import pandas as pd
import numpy as np


model = Ridge(alpha=10.0, max_iter=10000)


def data_preprocessing(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mapping = {"mediocre": 1, "moyenne": 2, "bonne": 3, "excellente": 4}
    df["qualite_cuisine_nbre"] = df["qualite_cuisine"].map(mapping)
    df["batiment_individuel"] = np.where(df["type_batiment"] == "individuelle", 1, 0)

    features = [
        "surf_hab",
        "qualite_materiau",
        "qualite_cuisine_nbre",
        "batiment_individuel",
        "n_pieces",
        "n_chambres_coucher",
        "surface_sous_sol",
        "surface_jardin",
        "n_garage_voitures",
    ]

    X = df[features].to_numpy()
    Y = df["prix"].to_numpy()

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    return X_poly, Y


def model_fit(X: np.ndarray, Y: np.ndarray):
    model.fit(X, Y)


def model_predict(X: np.ndarray) -> np.ndarray:
    return model.predict(X)
