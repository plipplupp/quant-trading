# backtest_ranking.py
import os
import math
import joblib
import pandas as pd
from config import PathsConfig, BacktestConfig, TargetConfig
from utils import (
    calculate_brokerage_fee,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)


def _get_feature_cols(df):
    return [c for c in df.columns if c not in ['date', 'ticker',
                                               'target_regression',
                                               'target_binary',
                                               'target_rank']]


def _extract_price(val):
    """Säker extraktion av pris oavsett scalar/Series."""
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    return float(val)


def check_data_quality(df):
    """Kontrollera datan för orealistiska värden per ticker"""
    print("\n--- Datakvalitetskontroll ---")

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date')

        # Kontrollera stora kurshopp
        ticker_df['price_change'] = ticker_df['adj_close'].pct_change()
        extreme_changes = ticker_df[abs(ticker_df['price_change']) > 0.5]  # >50% på en dag
        if not extreme_changes.empty:
            print(f"⚠️ {ticker}: Extrema prisförändringar funna:")
            for _, row in extreme_changes.head(3).iterrows():
                print(f"   {row['date'].date()}: {row['price_change']*100:+.1f}% (pris: {row['adj_close']:.2f})")

        # Kontrollera suspekta låga priser
        very_cheap = ticker_df[ticker_df['adj_close'] < 5.0]
        if not very_cheap.empty:
            print(f"⚠️ {ticker}: Mycket låga priser under 5 kr:")
            for _, row in very_cheap.head(3).iterrows():
                print(f"   {row['date'].date()}: {row['adj_close']:.2f} kr")


def backtest_ranking():
    print("\n--- Backtest Ranking ---")
    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl")
    if not os.path.exists(model_path):
        print("Ingen ranking-modell hittades. Kör train_models först.")
        return

    # --- Ladda data och modell ---
    df = pd.read_parquet(os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet"))
    model = joblib.load(model_path)

    features = _get_feature_cols(df)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    X = df[features].fillna(0)
    df['predicted_score'] = model.predict(X)

    # --- Datakvalitetskontroll ---
    check_data_quality(df)

    # --- Parametrar ---
    top_n = getattr(TargetConfig, 'RANK_TOP_N', None)
    top_pct = getattr(TargetConfig, 'RANK_TOP_PCT', None)
    rebalance_days = getattr(TargetConfig, 'RANK_REBALANCE_DAYS', 5)

    ticker_dfs = {t: g.set_index('date').sort_index() for t, g in df.groupby('ticker')}
    all_dates = sorted(df['date'].unique())
    date_to_group = {d: df[df['date'] == d] for d in all_dates}

    positions = {}
    cash = BacktestConfig.INITIAL_CAPITAL
    trade_log, daily_vals = [], []
    port_value = cash  # initialt värde

    for i, date in enumerate(all_dates):
        g = date_to_group[date]

        # --- Rebalancering ---
        if i % rebalance_days == 0:
            if top_n is not None:
                selected = g.sort_values('predicted_score', ascending=False).head(top_n)
            elif top_pct is not None:
                cutoff = g['predicted_score'].quantile(1 - top_pct)
                selected = g[g['predicted_score'] >= cutoff]
            else:
                selected = g.sort_values('predicted_score', ascending=False).head(10)

            selected_tickers = set(selected['ticker'].tolist())

            # --- Sälj utgående innehav ---
            for t, pos in list(positions.items()):
                if t not in selected_tickers:
                    tdf = ticker_dfs[t]
                    if date in tdf.index:
                        price = _extract_price(tdf.loc[date, 'adj_close'])
                        sale_value = pos['shares'] * price
                        fee = calculate_brokerage_fee(sale_value,
                                                      BacktestConfig.BROKERAGE_FIXED_FEE,
                                                      BacktestConfig.BROKERAGE_PERCENTAGE)
                        cash += sale_value - fee
                        trade_log.append({
                            'date': date, 'action': 'SELL', 'ticker': t,
                            'price': price, 'shares': pos['shares'], 'fee': fee,
                            'cash_after': cash, 'reason': 'REBALANCE_OUT'
                        })
                        positions.pop(t)

            # --- Köp nya innehav ---
            not_held = [t for t in selected_tickers if t not in positions]
            if not_held:
                capital_to_use = cash * BacktestConfig.TRADE_ALLOCATION
                capital_per = capital_to_use / len(not_held)

                for t in not_held:
                    tdf = ticker_dfs[t]
                    if date not in tdf.index:
                        continue
                    price = _extract_price(tdf.loc[date, 'adj_close'])
                    if price <= 0:
                        continue
                    shares = math.floor(capital_per / price)
                    if shares <= 0:
                        continue

                    buy_cost = shares * price
                    fee = calculate_brokerage_fee(buy_cost,
                                                  BacktestConfig.BROKERAGE_FIXED_FEE,
                                                  BacktestConfig.BROKERAGE_PERCENTAGE)
                    total_cost = buy_cost + fee

                    if total_cost <= cash:
                        cash -= total_cost
                        positions[t] = {'shares': shares, 'purchase_price': price}
                        trade_log.append({
                            'date': date, 'action': 'BUY', 'ticker': t,
                            'price': price, 'shares': shares, 'fee': fee,
                            'cash_after': cash, 'reason': 'REBALANCE_IN'
                        })

            # --- Stop-loss ---
            for t, pos in list(positions.items()):
                tdf = ticker_dfs[t]
                if date not in tdf.index:
                    continue
                price_today = _extract_price(tdf.loc[date, 'adj_close'])
                if price_today <= pos['purchase_price'] * (1 - BacktestConfig.STOP_LOSS_PCT):
                    sale_value = pos['shares'] * price_today
                    fee = calculate_brokerage_fee(sale_value,
                                                  BacktestConfig.BROKERAGE_FIXED_FEE,
                                                  BacktestConfig.BROKERAGE_PERCENTAGE)
                    cash += sale_value - fee
                    trade_log.append({
                        'date': date, 'action': 'SELL', 'ticker': t,
                        'price': price_today, 'shares': pos['shares'], 'fee': fee,
                        'cash_after': cash, 'reason': 'STOP-LOSS'
                    })
                    positions.pop(t)

        # --- Uppdatera portföljvärde ---
        port_value = cash
        for t, pos in positions.items():
            tdf = ticker_dfs[t]
            idx = tdf.index.searchsorted(date, side="right") - 1
            if idx >= 0:
                last_price = _extract_price(tdf.iloc[idx]['adj_close'])
                port_value += pos['shares'] * last_price

        daily_vals.append({'date': date, 'portfolio_value': port_value})

    # --- Summera resultat ---
    trades_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_vals)
    if daily_df.empty:
        print("⚠️ Inga dagliga värden genererades.")
        return

    daily_df.set_index('date', inplace=True)
    final_value = float(daily_df['portfolio_value'].iloc[-1])
    total_profit = final_value - BacktestConfig.INITIAL_CAPITAL
    total_fees = trades_df['fee'].sum() if not trades_df.empty else 0.0
    total_trades = len(trades_df)
    daily_returns = daily_df['portfolio_value'].pct_change().dropna()

    sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0
    sortino = calculate_sortino_ratio(daily_returns) if len(daily_returns) > 0 else 0.0
    maxdd = calculate_max_drawdown(daily_df['portfolio_value'])

    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    trades_out = os.path.join(PathsConfig.RESULTS_DIR, "ranking_trades.csv")
    daily_out = os.path.join(PathsConfig.RESULTS_DIR, "ranking_daily.csv")
    format_trades_csv(trades_df, trades_out)
    daily_df.to_csv(daily_out)

    print("\n--- Ranking backtest summary ---")
    print(f"Slutkapital: {final_value:,.2f} kr")
    print(f"Total vinst: {total_profit:,.2f} kr")
    print(f"Total courtageavgift: {total_fees:,.2f} kr")
    print(f"Totalt antal transaktioner: {total_trades}")
    print(f"Sharpe (år): {sharpe:.2f}, Sortino (år): {sortino:.2f}, MaxDD: {maxdd:.2%}")
    print(f"Sparade trades -> {trades_out}, daglig portfölj -> {daily_out}")


def format_trades_csv(trades_df, output_path):
    """Formaterar trades DataFrame med färre decimaler innan CSV-export"""
    if trades_df.empty:
        return

    formatted_df = trades_df.copy()
    formatted_df['price'] = formatted_df['price'].round(2)
    formatted_df['shares'] = formatted_df['shares'].astype(int)
    formatted_df['fee'] = formatted_df['fee'].round(0).astype(int)
    formatted_df['cash_after'] = formatted_df['cash_after'].round(0).astype(int)
    formatted_df.to_csv(output_path, index=False)
    return formatted_df
