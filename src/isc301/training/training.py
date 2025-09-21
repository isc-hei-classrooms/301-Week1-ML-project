from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train(df):
    # --- split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=["prix"])
    y_train = train_df["prix"]
    X_test  = test_df.drop(columns=["prix"])
    y_test  = test_df["prix"]

    # --- preprocessors
    categorical_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
    numeric_cols     = X_train.select_dtypes(exclude=['object','category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    # --- model & pipeline
    elastic = ElasticNet(max_iter=10000, random_state=42)
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', elastic)])

    # --- CV & grid
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'model__alpha':    np.logspace(-3, 2, 10),
        'model__l1_ratio': np.linspace(0.05, 0.95, 10)
    }

    # multi-métriques; on refit sur la meilleure R²
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring={'r2':'r2', 'neg_mae':'neg_mean_absolute_error', 'neg_rmse':'neg_root_mean_squared_error'},
        refit='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    # --- fit grid
    gscv.fit(X_train, y_train)

    print("Best params:", gscv.best_params_)
    print(f"Best CV R²: {gscv.best_score_:.3f}")

    # --- évaluation test
    best_model = gscv.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse  = root_mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("\nTest metrics (Elastic Net with GridSearchCV):")
    print(f"  R²   : {r2:.3f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")

    # --- tableau des coefficients
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    coefs = best_model.named_steps['model'].coef_
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs, 'abs_coef': np.abs(coefs)}) \
                .sort_values('abs_coef', ascending=False)
    print("\nTop features:\n", coef_df.head(40))

    return best_model
          
def final_test(best_model, df):
    X_full = df.drop(columns=["prix"])
    y_full = df["prix"]

    y_pred_full = best_model.predict(X_full)

    rmse_full = root_mean_squared_error(y_full, y_pred_full)
    mae_full  = mean_absolute_error(y_full, y_pred_full)
    r2_full   = r2_score(y_full, y_pred_full)

    residuals = y_full - y_pred_full

    plt.figure(figsize=(7,4))
    colors = np.where(df["is_expensive"], "red", "blue")
    plt.scatter(y_full, residuals, c=colors, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valeurs réelles (prix)")
    plt.ylabel("Résidus (y - ŷ)")
    plt.title("Résidus en fonction du prix réel")
    plt.show()


    print("\nEvaluation sur tout le dataset :")
    print(f"  R²   : {r2_full:.3f}")
    print(f"  RMSE : {rmse_full:.2f}")
    print(f"  MAE  : {mae_full:.2f}")