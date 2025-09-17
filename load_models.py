import os
import joblib
from config import PathsConfig

def load_models():
    """
    Laddar regression, binary och ranking-modeller från disk.
    Returnerar en dict med nycklarna: regression, binary, ranking.
    """
    models = {}
    paths = {
        "regression": os.path.join(PathsConfig.MODELS_DIR, "model_regression.pkl"),
        "binary": os.path.join(PathsConfig.MODELS_DIR, "model_binary.pkl"),
        "ranking": os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl"),
    }

    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"✅ Laddade {name}-modell från {path}")
        else:
            print(f"⚠️ Modell saknas: {path}. Kör train_models först.")

    return models
