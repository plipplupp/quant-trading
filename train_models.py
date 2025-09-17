import os
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from config import PathsConfig, TrainingConfig


def train_models():
    print("\n--- Steg: Tränar modeller ---")

    # --- Läs in data ---
    data_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(data_path):
        print("Fel: Kör generate_targets först.")
        return

    df = pd.read_parquet(data_path)

    features = [c for c in df.columns if c not in [
        'date', 'ticker',
        'target_regression', 'target_binary', 'target_rank'
    ]]

    os.makedirs(PathsConfig.MODELS_DIR, exist_ok=True)

    # --- Gemensam tidsserie-split ---
    tscv = TimeSeriesSplit(n_splits=TrainingConfig.CV_SPLITS)

    # =========================================================
    # Regression
    # =========================================================
    print("\n=== Tränar Regression-modell ===")
    df_reg = df.dropna(subset=['target_regression'])
    X, y = df_reg[features], df_reg['target_regression']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model_reg = XGBRegressor(
        random_state=TrainingConfig.RANDOM_STATE,
        tree_method="hist",
        device="cuda" if TrainingConfig.USE_GPU else "cpu"
    )

    pipe_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_reg)
    ])

    param_grid_reg = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.7, 0.9, 1.0]
    }

    search_reg = RandomizedSearchCV(
        pipe_reg, param_grid_reg,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv,
        n_jobs=TrainingConfig.N_JOBS,
        scoring='r2', verbose=1
    )
    search_reg.fit(X_train, y_train)

    y_pred = search_reg.predict(X_test)
    print(f"Regression R² (holdout): {r2_score(y_test, y_pred):.3f}")
    print(f"Regression RMSE (holdout): {root_mean_squared_error(y_test, y_pred):.4f}")

    joblib.dump(search_reg.best_estimator_,
                os.path.join(PathsConfig.MODELS_DIR, "model_regression.pkl"))
    print("✅ Sparade regression-modell")

    # =========================================================
    # Binary
    # =========================================================
    print("\n=== Tränar Binary-modell ===")
    df_bin = df.dropna(subset=['target_binary'])
    X, y = df_bin[features], df_bin['target_binary']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model_bin = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=TrainingConfig.RANDOM_STATE,
        tree_method="hist",
        device="cuda" if TrainingConfig.USE_GPU else "cpu"
    )

    pipe_bin = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_bin)
    ])

    param_grid_bin = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.7, 0.9, 1.0]
    }

    search_bin = RandomizedSearchCV(
        pipe_bin, param_grid_bin,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv,
        n_jobs=TrainingConfig.N_JOBS,
        scoring='accuracy', verbose=1
    )
    search_bin.fit(X_train, y_train)

    y_pred = search_bin.predict(X_test)
    print(f"Binary Accuracy (holdout): {accuracy_score(y_test, y_pred):.3f}")

    joblib.dump(search_bin.best_estimator_,
                os.path.join(PathsConfig.MODELS_DIR, "model_binary.pkl"))
    print("✅ Sparade binary-modell")

    # =========================================================
    # Ranking (Regression för avkastning)
    # =========================================================
    print("\n=== Tränar Ranking-modell ===")
    df_rank = df.dropna(subset=['target_rank'])
    X, y = df_rank[features], df_rank['target_rank']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model_rank = XGBRegressor(
        random_state=TrainingConfig.RANDOM_STATE,
        tree_method="hist",
        device="cuda" if TrainingConfig.USE_GPU else "cpu"
    )

    pipe_rank = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_rank)
    ])

    param_grid_rank = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.7, 0.9, 1.0]
    }

    search_rank = RandomizedSearchCV(
        pipe_rank, param_grid_rank,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv,
        n_jobs=TrainingConfig.N_JOBS,
        scoring='r2', verbose=1
    )
    search_rank.fit(X_train, y_train)

    y_pred = search_rank.predict(X_test)
    print(f"Ranking R² (holdout): {r2_score(y_test, y_pred):.3f}")
    print(f"Ranking RMSE (holdout): {root_mean_squared_error(y_test, y_pred):.4f}")

    joblib.dump(search_rank.best_estimator_,
                os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl"))
    print("✅ Sparade ranking-modell")


if __name__ == "__main__":
    train_models()
