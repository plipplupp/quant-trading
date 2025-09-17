import os
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from config import PathsConfig, TrainingConfig
import numpy as np


def _train_single_model(model_type, df, features, target_col):
    """
    Hjälpfunktion för att träna en enskild modell (Regression, Binary, eller Ranking).
    Använder nu TimeSeriesSplit för både träning och en slutgiltig holdout-utvärdering.
    """
    print(f"\n=== Tränar {model_type}-modell ===")
    
    # 1. Förbered data
    df_model = df.dropna(subset=[target_col])
    X, y = df_model[features], df_model[target_col]

    if X.empty:
        print(f"Ingen data att träna på för {model_type}-modellen efter rensning. Avbryter.")
        return

    # 2. Definiera en TimeSeriesSplit för både CV och en slutgiltig holdout-split
    # Vi använder den sista splitten som vår train/test-uppdelning för utvärdering
    tscv_outer = TimeSeriesSplit(n_splits=TrainingConfig.CV_SPLITS + 1)
    all_splits = list(tscv_outer.split(X))
    train_indices, test_indices = all_splits[-1] # Sista splitten är vår holdout

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    # Den inre korsvalideringen för RandomizedSearch använder en egen TimeSeriesSplit
    tscv_inner = TimeSeriesSplit(n_splits=TrainingConfig.CV_SPLITS)

    # 3. Definiera modell och pipeline
    is_classifier = model_type == 'Binary'
    
    common_xgb_params = {
        'random_state': TrainingConfig.RANDOM_STATE,
        'device': "cuda" if TrainingConfig.USE_GPU else "cpu"
    }

    if is_classifier:
        model = XGBClassifier(**common_xgb_params, eval_metric='logloss', use_label_encoder=False)
    else:
        model = XGBRegressor(**common_xgb_params)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # 4. Definiera sök-parametrar och kör RandomizedSearchCV
    param_grid = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv_inner,
        n_jobs=TrainingConfig.N_JOBS,
        scoring='r2' if not is_classifier else 'accuracy',
        verbose=1
    )
    search.fit(X_train, y_train)

    # 5. Utvärdera och spara bästa modellen på holdout-setet
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if is_classifier:
        score = accuracy_score(y_test, y_pred)
        print(f"{model_type} Accuracy (holdout): {score:.3f}")
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"{model_type} R² (holdout): {r2:.3f}")
        print(f"{model_type} RMSE (holdout): {rmse:.4f}")

    model_filename = f"model_{model_type.lower()}.pkl"
    joblib.dump(best_model, os.path.join(PathsConfig.MODELS_DIR, model_filename))
    print(f"✅ Sparade {model_type}-modell")


def train_models():
    print("\n--- Steg: Tränar modeller ---")

    # --- Läs in data ---
    data_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(data_path):
        print("Fel: Kör generate_targets först.")
        return

    df = pd.read_parquet(data_path)

    ### FIX: Återställer din ursprungliga, korrekta metod för att välja features
    features = [c for c in df.columns if c not in [
        'date', 'ticker',
        'target_regression', 'target_binary', 'target_rank'
    ]]

    # Säkerställ att alla feature-kolumner är numeriska
    # Detta är en extra säkerhetsåtgärd
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    # Ersätt eventuella oändliga värden med NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Ta bort rader där NÅGON feature är NaN, detta är det säkraste för träning
    df.dropna(subset=features, inplace=True)


    print(f"Hittade {len(features)} features att träna på.")
    if not features:
        print("Fel: Inga feature-kolumner hittades. Kontrollera din datakälla och kolumnnamn.")
        return

    os.makedirs(PathsConfig.MODELS_DIR, exist_ok=True)
    
    # Anropa hjälpfunktionen för varje modell
    _train_single_model('Regression', df, features, 'target_regression')
    _train_single_model('Binary', df, features, 'target_binary')
    _train_single_model('Ranking', df, features, 'target_rank')


if __name__ == "__main__":
    train_models()