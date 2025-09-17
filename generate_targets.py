import sqlite3
import pandas as pd
import numpy as np
from config import DatabaseConfig, PathsConfig, TargetConfig
import os
import shutil

def generate_targets():
    print("\n--- Steg: Genererar målvariabler ---")
    conn = sqlite3.connect(DatabaseConfig.DB_NAME)

    df = pd.read_sql("SELECT * FROM stocks_prepared", conn)
    conn.close()

    if df.empty:
        print("Fel: 'stocks_prepared' är tom. Kör data_pipeline först.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['ticker', 'date'], inplace=True)

    all_targets = []

    for ticker, g in df.groupby('ticker'):
        g = g.copy()

        # Regression target
        g['target_regression'] = g['adj_close'].shift(-TargetConfig.REGRESSION_DAYS) / g['adj_close'] - 1

        # Binary vs OMX
        if 'omx_close' in g.columns:
            omx_future = g['omx_close'].shift(-TargetConfig.BINARY_DAYS) / g['omx_close'] - 1
            stock_future = g['adj_close'].shift(-TargetConfig.BINARY_DAYS) / g['adj_close'] - 1
            g['target_binary'] = (stock_future > omx_future).astype(int)
        else:
            g['target_binary'] = np.nan

        # Ranking score
        g['target_rank'] = g['adj_close'].shift(-TargetConfig.RANKING_DAYS) / g['adj_close'] - 1

        all_targets.append(g)

    df_targets = pd.concat(all_targets, ignore_index=True)

    # Se till att rätt mapp används
    os.makedirs(PathsConfig.TARGETS_DIR, exist_ok=True)
    out_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    df_targets.to_parquet(out_path, index=False)

    print(f"Sparade {len(df_targets):,} rader med målvariabler till {out_path}")

    # --- Rensa bort eventuell felaktig mapp ---
    wrong_path = os.path.join(PathsConfig.DATA_DIR, "stocks_with_targets")
    if os.path.exists(wrong_path):
        print(f"Tar bort gammal mapp: {wrong_path}")
        shutil.rmtree(wrong_path)
