import os
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from config import PathsConfig, TrainingConfig

def _train_single_model(model_type, df, features, target_col):
    """
    Hjälpfunktion för att träna en enskild modell (Regression, Binary, eller Ranking).
    Denna funktion innehåller all logik som tidigare upprepades.
    """
    print(f"\n=== Tränar {model_type}-modell ===")
    
    # 1. Förbered data
    df_model = df.dropna(subset=[target_col])
    X, y = df_model[features], df_model[target_col]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 2. Definiera modell och pipeline
    is_classifier = model_type == 'Binary'
    
    ### FIX: Ändra tree_method till 'gpu_hist' för att aktivera den optimerade GPU-algoritmen
    common_xgb_params = {
        'random_state': TrainingConfig.RANDOM_STATE,
        'tree_method': "gpu_hist" if TrainingConfig.USE_GPU else "hist",
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

    # 3. Definiera sök-parametrar och kör RandomizedSearchCV
    param_grid = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0] # Ny parameter att testa
    }

    tscv = TimeSeriesSplit(n_splits=TrainingConfig.CV_SPLITS)
    
    search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv,
        n_jobs=TrainingConfig.N_JOBS,
        scoring='r2' if not is_classifier else 'accuracy',
        verbose=1
    )
    search.fit(X_train, y_train)

    # 4. Utvärdera och spara bästa modellen
    y_pred = search.predict(X_test)
    
    if is_classifier:
        score = accuracy_score(y_test, y_pred)
        print(f"{model_type} Accuracy (holdout): {score:.3f}")
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"{model_type} R² (holdout): {r2:.3f}")
        print(f"{model_type} RMSE (holdout): {rmse:.4f}")

    model_filename = f"model_{model_type.lower()}.pkl"
    joblib.dump(search.best_estimator_, os.path.join(PathsConfig.MODELS_DIR, model_filename))
    print(f"✅ Sparade {model_type}-modell")


def train_models():
    print("\n--- Steg: Tränar modeller ---")

    # --- Läs in data ---
    data_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(data_path):
        print("Fel: Kör generate_targets först.")
        return

    df = pd.read_parquet(data_path)

    features = [c for c in df.columns if c.startswith('feature_')]

    os.makedirs(PathsConfig.MODELS_DIR, exist_ok=True)
    
    # Anropa hjälpfunktionen för varje modell
    _train_single_model('Regression', df, features, 'target_regression')
    _train_single_model('Binary', df, features, 'target_binary')
    _train_single_model('Ranking', df, features, 'target_rank')


if __name__ == "__main__":
    train_models()