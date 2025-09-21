import pandas as pd

def preprocessing(df):
    qualite_map = {"mediocre": 1, "moyenne": 2, "bonne": 3, "excellente": 4}
    df["qualite_cuisine"] = df["qualite_cuisine"].map(qualite_map)

    df = df[df["n_pieces"] <= 11]
    df = df[df["n_cheminees"] <= 2]
    df = df[df["n_chambres_coucher"] < 5]
    df = df[df["prix"] < 500000]
    df = df[df["n_garage_voitures"] <= 3]
    df = df[df["surface_sous_sol"] < 2500]
    df = df[df["n_toilettes"] <= 1]

    # if expensive if qualite_cuisine >=3 and qualite_materiau >= 8 and n_garage_voitures >= 3
    df["is_expensive"] = ((df["qualite_cuisine"] >= 3) & (df["qualite_materiau"] >= 8) & (df["n_garage_voitures"] >= 3)).astype(int)

    df["expensive_surface"] = df["is_expensive"] * df["surf_hab"]

    # add a column which is surf_hab / n_pieces
    df["surf_hab_per_piece"] = df["surf_hab"] / df["n_pieces"]

    df.drop(columns=["annee_vente", "qualite_globale", "n_cuisines", "surface_jardin", "n_pieces"], inplace=True)

    return df