import os
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from config import PathsConfig, TrainingConfig

def _train_single_model(model_type, X_train, y_train, X_test, y_test):
    """
    En hjälpfunktion för att träna, utvärdera och spara en enskild modell.
    Detta minskar kodrepetition.
    """
    print(f"\n=== Tränar {model_type}-modell ===")

    # Välj modell och hyperparametrar baserat på typ
    if model_type in ['Regression', 'Ranking']:
        model = XGBRegressor(
            random_state=TrainingConfig.RANDOM_STATE,
            tree_method="hist",
            device="cuda" if TrainingConfig.USE_GPU else "cpu"
        )
        scoring = 'r2'
        param_grid = {
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [3, 5, 7, 9],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.7, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.9, 1.0]
        }
    elif model_type == 'Binary':
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=TrainingConfig.RANDOM_STATE,
            tree_method="hist",
            device="cuda" if TrainingConfig.USE_GPU else "cpu"
        )
        scoring = 'accuracy'
        # Använder samma grid som regression för enkelhetens skull, kan anpassas
        param_grid = {
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [3, 5, 7, 9],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.7, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.9, 1.0]
        }
    else:
        raise ValueError("Okänd modelltyp")

    # Skapa pipeline och tidsserie-split
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    tscv = TimeSeriesSplit(n_splits=TrainingConfig.CV_SPLITS)

    # Kör RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=TrainingConfig.RANDOM_SEARCH_ITERS,
        cv=tscv,
        n_jobs=TrainingConfig.N_JOBS,
        scoring=scoring,
        verbose=1,
        random_state=TrainingConfig.RANDOM_STATE
    )
    search.fit(X_train, y_train)

    # Utvärdera på holdout-set och skriv ut resultat
    y_pred = search.predict(X_test)
    if model_type in ['Regression', 'Ranking']:
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"{model_type} R² (holdout): {r2:.3f}")
        print(f"{model_type} RMSE (holdout): {rmse:.4f}")
    elif model_type == 'Binary':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_type} Accuracy (holdout): {accuracy:.3f}")

    # Spara den bästa modellen
    model_filename = f"model_{model_type.lower()}.pkl"
    joblib.dump(search.best_estimator_, os.path.join(PathsConfig.MODELS_DIR, model_filename))
    print(f"✅ Sparade {model_type.lower()}-modell")


def train_models():
    print("\n--- Steg: Tränar modeller ---")

    # --- Läs in data ---
    data_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(data_path):
        print(f"Fel: Filen {data_path} hittades inte. Kör generate_targets först.")
        return

    df = pd.read_parquet(data_path)

    # Definiera vilka kolumner som är features
    features = [c for c in df.columns if c.startswith('feature_')]
    
    # Säkerställ att mappen för modeller finns
    os.makedirs(PathsConfig.MODELS_DIR, exist_ok=True)

    # --- Modellträning ---
    # Notera den KORREKTA ordningen:
    # 1. Välj target och rensa bort NaN-värden.
    # 2. Definiera X och y FRÅN DEN RENA DATAN.
    # 3. Dela upp X och y i train/test.
    # 4. Skicka den färdigdelade datan till hjälpfunktionen.

    # Regression
    df_reg = df.dropna(subset=['target_regression'])
    if not df_reg.empty:
        X, y = df_reg[features], df_reg['target_regression']
        split_idx = int(len(X) * TrainingConfig.TRAIN_SPLIT_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        _train_single_model('Regression', X_train, y_train, X_test, y_test)
    else:
        print("\nSkippar Regression-modell: Ingen data efter rensning av NaN.")

    # Binary
    df_bin = df.dropna(subset=['target_binary'])
    if not df_bin.empty:
        X, y = df_bin[features], df_bin['target_binary']
        split_idx = int(len(X) * TrainingConfig.TRAIN_SPLIT_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        _train_single_model('Binary', X_train, y_train, X_test, y_test)
    else:
        print("\nSkippar Binary-modell: Ingen data efter rensning av NaN.")

    # Ranking
    df_rank = df.dropna(subset=['target_rank'])
    if not df_rank.empty:
        X, y = df_rank[features], df_rank['target_rank']
        split_idx = int(len(X) * TrainingConfig.TRAIN_SPLIT_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        _train_single_model('Ranking', X_train, y_train, X_test, y_test)
    else:
        print("\nSkippar Ranking-modell: Ingen data efter rensning av NaN.")


if __name__ == "__main__":
    train_models()