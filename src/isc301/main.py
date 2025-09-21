from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
lasso = Lasso(alpha=10.0, max_iter=10000)
col_names = ["qualite_cuisine", "surf_hab","n_pieces","qualite_globale","type_batiment","type_toit","n_garage_voitures","surface_sous_sol","surface_jardin"]

def data_preprocessing(df):
    mapping_cuisine = {
        "mediocre": 0,
        "moyenne": 1,
        "bonne": 2,
        "excellente": 3
    }
    mapping_toit = {
        "2 pans": 0,
        "4 pans": 1,
        "mansarde": 2,
        "plat": 3,
        "1 pans": 4
    }
    mapping_type_batiment = {
        "individuelle": 0,
        "milieu de rangée": 1,
        "bout de rangée": 2,
        "duplex": 3,
        "individuelle reconvertie": 4
    }
    df["qualite_cuisine"] = df["qualite_cuisine"].replace(mapping_cuisine)
    df["type_toit"] = df["type_toit"].replace(mapping_toit)
    df["type_batiment"] = df["type_batiment"].replace(mapping_type_batiment)
    df[col_names] = poly.fit_transform(df[col_names])
    return df [col_names],df["prix"]

def model_fit(X, Y):
    lasso.fit(X, Y)

def model_predict(X):
    return lasso.predict(X)