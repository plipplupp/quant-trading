# =============================================================================
# backtest_regression.py - Uppdaterad version
# =============================================================================

import os
import joblib
import pandas as pd
import numpy as np
from config import PathsConfig, BacktestConfig
from utils import simulate_engine, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

def _get_feature_cols(df):
    return [c for c in df.columns if c not in ['date', 'ticker',
                                               'target_regression',
                                               'target_binary',
                                               'target_rank']]

def backtest_regression():
    print("\n--- Backtest Regression (med säljsignaler) ---")
    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_regression.pkl")
    if not os.path.exists(model_path):
        print("Ingen regression-modell hittades. Kör train_models först.")
        return

    df = pd.read_parquet(os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet"))
    model = joblib.load(model_path)
    features = _get_feature_cols(df)

    # Gör prediktioner
    X = df[features].fillna(0)
    preds = model.predict(X)
    df = df.copy()
    df['predicted_return'] = preds

    # Generera signaler baserat på nya tröskelvärden
    def generate_regression_signal(pred_return):
        if pred_return > BacktestConfig.REGRESSION_THRESHOLD_BUY:
            return 1   # Köp
        elif pred_return < BacktestConfig.REGRESSION_THRESHOLD_SELL:
            return -1  # Sälj
        else:
            return 0   # Håll/neutral

    df['signal'] = df['predicted_return'].apply(generate_regression_signal)

    # Kontrollera signalfördelning
    signal_counts = df['signal'].value_counts().sort_index()
    print(f"Signal fördelning: {dict(signal_counts)}")
    print(f"Köp-signaler: {signal_counts.get(1, 0)}, Sälj-signaler: {signal_counts.get(-1, 0)}, Håll: {signal_counts.get(0, 0)}")

    # Kör backtest
    buy_signals_df = df[['date','ticker','signal']].copy()
    trades_df, daily_df = simulate_engine(df, buy_signals_df, BacktestConfig.INITIAL_CAPITAL,
                                          BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE,
                                          BacktestConfig.TRADE_ALLOCATION, BacktestConfig.STOP_LOSS_PCT)

    # Beräkna statistik
    if daily_df.empty:
        print("Inga dagliga värden genererades.")
        return
        
    daily_df.set_index('date', inplace=True)
    final_value = float(daily_df['portfolio_value'].iloc[-1])
    total_profit = final_value - BacktestConfig.INITIAL_CAPITAL
    total_fees = trades_df['fee'].sum() if not trades_df.empty else 0.0
    total_trades = len(trades_df)
    
    # Räkna köp vs sälj transaktioner
    buy_trades = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
    sell_trades = len(trades_df[trades_df['action'] == 'SELL']) if not trades_df.empty else 0

    daily_returns = daily_df['portfolio_value'].pct_change().dropna()
    sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns)>0 else 0.0
    sortino = calculate_sortino_ratio(daily_returns) if len(daily_returns)>0 else 0.0
    maxdd = calculate_max_drawdown(daily_df['portfolio_value'])

    # Spara resultat
    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    trades_out = os.path.join(PathsConfig.RESULTS_DIR, "regression_trades.csv")
    daily_out = os.path.join(PathsConfig.RESULTS_DIR, "regression_daily.csv")

    # Format kolumner i trades_df
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['price'] = trades_df['price'].map(lambda x: f"{x:.2f}")
        trades_df['fee'] = trades_df['fee'].map(lambda x: f"{x:.0f}")
        trades_df['cash_after'] = trades_df['cash_after'].map(lambda x: f"{x:,.0f}".replace(",", ""))

    trades_df.to_csv(trades_out, index=False)
    daily_df.to_csv(daily_out)

    print("\n--- Regression backtest summary ---")
    print(f"Slutkapital: {final_value:,.0f} kr".replace(",", " "))
    print(f"Total vinst: {total_profit:,.0f} kr ({total_profit/BacktestConfig.INITIAL_CAPITAL*100:.0f}%)".replace(",", " "))
    print(f"Total courtageavgift: {total_fees:,.0f} kr".replace(",", " "))
    print(f"Totalt antal transaktioner: {total_trades} (Köp: {buy_trades}, Sälj: {sell_trades})")
    print(f"Sharpe (år): {sharpe:.2f}, Sortino (år): {sortino:.2f}, MaxDD: {maxdd:.2%}")
    print(f"Tröskelvärden: Köp > {BacktestConfig.REGRESSION_THRESHOLD_BUY}, Sälj < {BacktestConfig.REGRESSION_THRESHOLD_SELL}")
    print(f"Sparade trades -> {trades_out}, daglig portfölj -> {daily_out}")

    # ============================
    # Skapa signal-CSV för senaste datumet
    # ============================
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].copy()

    # Mappa signaler till text
    signal_map = {1: "Köp", -1: "Sälj", 0: "Neutral"}
    latest_df['recommendation'] = latest_df['signal'].map(signal_map)

    # Sortera: Köp först, sedan Sälj, sist Neutral
    latest_df = latest_df.sort_values(by='predicted_return', ascending=False)

    # Konvertera predicted_return till procent och format
    latest_df['predicted_return'] = latest_df['predicted_return'] * 100

    # Döp om kolumnen
    latest_df.rename(columns={'predicted_return': 'predicted_return_[%]'}, inplace=True)

    # Välj kolumner att spara
    signals_out = os.path.join(PathsConfig.RESULTS_DIR, "regression_signals_today.csv")
    latest_df[['ticker', 'recommendation', 'predicted_return_[%]']].to_csv(signals_out, index=False)

    print(f"\nDagens signaler ({latest_date}): sparade till {signals_out}")

if __name__ == "__main__":
    backtest_regression()