import os
import math
import traceback
import pandas as pd
import joblib
from config import PathsConfig, BacktestConfig, TargetConfig
from utils import calculate_brokerage_fee

def _get_feature_cols(df):
    """Hämta alla feature-kolumner (exkl. datum, ticker och targets)."""
    return [c for c in df.columns if c not in [
        'date', 'ticker',
        'target_regression',
        'target_binary',
        'target_rank'
    ]]

def _extract_price(val):
    """Säker extraktion av pris oavsett scalar/Series/pandas scalar."""
    try:
        if hasattr(val, 'iloc'):
            return float(val.iloc[0])
        return float(val)
    except Exception:
        # sista utväg: konvertera via pandas
        return float(pd.Series([val]).astype(float).iloc[0])

def debug_backtest_ranking():
    print("\n--- Debug Backtest Ranking ---")

    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl")
    if not os.path.exists(model_path):
        print("Ingen ranking-modell hittades. Kör train_models först.")
        return

    # Läs data + modell
    df_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(df_path):
        print(f"Ingen datafil hittades: {df_path}")
        return

    df = pd.read_parquet(df_path)
    if df.empty:
        print("Datafilen är tom.")
        return

    # normalisera datumkolumn tidigt
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # info
    n_tickers = df['ticker'].nunique()
    date_min, date_max = df['date'].min(), df['date'].max()
    print(f"Data: {n_tickers} tickers, datum {date_min.date()} -> {date_max.date()}, rader: {len(df):,}")

    # features
    features = _get_feature_cols(df)
    print(f"Antal features: {len(features)}")

    # ladda modell
    try:
        model = joblib.load(model_path)
        print(f"Laddade modell från {model_path} (typ: {type(model)})")
    except Exception as e:
        print("Fel vid inläsning av modell:")
        traceback.print_exc()
        return

    # försök predict på några rader för att upptäcka mismatch tidigt
    X = df[features].fillna(0)
    try:
        # snabb testprediktion
        _ = model.predict(X.iloc[:5])
    except Exception as e:
        print("Fel vid model.predict — troligen feature mismatch eller formateringsproblem.")
        print("Exception:")
        traceback.print_exc()
        # Skriv ut diagnosinfo
        print("Tillgängliga kolumner i X (första 20):", list(X.columns)[:20])
        # Om model är pipeline, försök visa input-feature-namn från eventuell ColumnTransformer (om möjligt)
        try:
            if hasattr(model, 'named_steps'):
                print("Modellens steg:", model.named_steps.keys())
        except Exception:
            pass
        return

    # beräkna predicted_score
    df['predicted_score'] = model.predict(X)

    # skapa per-ticker index med normaliserat datum för pålitlig åtkomst
    ticker_dfs = {}
    for t, g in df.groupby('ticker'):
        g2 = g.copy()
        g2['date'] = pd.to_datetime(g2['date']).dt.normalize()
        g2 = g2.set_index('date').sort_index()
        ticker_dfs[t] = g2

    # alla handelsdagar (normaliserade)
    all_dates = sorted(df['date'].dt.normalize().unique())
    date_to_group = {d: df[df['date'] == d] for d in all_dates}

    # state
    positions = {}
    cash = float(BacktestConfig.INITIAL_CAPITAL)
    daily_logs = []
    rebalance_days = getattr(TargetConfig, 'RANK_REBALANCE_DAYS', 5)
    top_n = getattr(TargetConfig, 'RANK_TOP_N', None)

    for i, date in enumerate(all_dates):
        g = date_to_group[date]
        trades_today = []

        # Rebalance logic (samma-day buys/sells here; debug-version keeps it simple)
        if i % rebalance_days == 0:
            selected = g.sort_values('predicted_score', ascending=False).head(top_n or 10)
            selected_tickers = set(selected['ticker'].tolist())

            # SELL holdings not in selected
            for t, pos in list(positions.items()):
                if t not in selected_tickers:
                    tdf = ticker_dfs.get(t)
                    if tdf is None or date not in tdf.index:
                        continue
                    price = _extract_price(tdf.loc[date, 'adj_close'])
                    sale_value = pos['shares'] * price
                    fee = calculate_brokerage_fee(sale_value,
                                                  BacktestConfig.BROKERAGE_FIXED_FEE,
                                                  BacktestConfig.BROKERAGE_PERCENTAGE)
                    cash += sale_value - fee
                    trades_today.append(f"SELL {t} {pos['shares']} @ {price:.2f} fee {fee:.2f}")
                    positions.pop(t)

            # BUY new selected (use allocation of current cash)
            not_held = [t for t in selected_tickers if t not in positions]
            if not_held:
                capital_to_use = cash * float(BacktestConfig.TRADE_ALLOCATION)
                capital_per = capital_to_use / len(not_held)

                for t in not_held:
                    tdf = ticker_dfs.get(t)
                    if tdf is None or date not in tdf.index:
                        continue
                    price = _extract_price(tdf.loc[date, 'adj_close'])
                    if price <= 0:
                        continue
                    shares = math.floor(capital_per / price)
                    # justera shares så buy_cost+fee <= cash
                    while shares > 0:
                        buy_cost = shares * price
                        fee = calculate_brokerage_fee(buy_cost,
                                                      BacktestConfig.BROKERAGE_FIXED_FEE,
                                                      BacktestConfig.BROKERAGE_PERCENTAGE)
                        if buy_cost + fee <= cash:
                            break
                        shares -= 1
                    if shares <= 0:
                        continue
                    buy_cost = shares * price
                    fee = calculate_brokerage_fee(buy_cost,
                                                  BacktestConfig.BROKERAGE_FIXED_FEE,
                                                  BacktestConfig.BROKERAGE_PERCENTAGE)
                    total_cost = buy_cost + fee
                    cash -= total_cost
                    positions[t] = {'shares': shares, 'purchase_price': price}
                    trades_today.append(f"BUY {t} {shares} @ {price:.2f} fee {fee:.2f}")

        # mark-to-market
        port_value = cash
        positions_snapshot = {}
        for t, pos in positions.items():
            tdf = ticker_dfs.get(t)
            if tdf is None:
                last_price = pos['purchase_price']
            else:
                if date in tdf.index:
                    last_price = _extract_price(tdf.loc[date, 'adj_close'])
                else:
                    idx = tdf.index.searchsorted(date, side="right") - 1
                    if idx >= 0:
                        last_price = _extract_price(tdf.iloc[idx]['adj_close'])
                    else:
                        last_price = pos['purchase_price']
            val = pos['shares'] * last_price
            positions_snapshot[t] = val
            port_value += val

        daily_logs.append({
            'date': date,
            'cash': int(round(cash)),
            'portfolio_value': int(round(port_value)),
            'positions': "; ".join([f"{t}:{int(round(v))}" for t, v in positions_snapshot.items()]),
            'trades': "; ".join(trades_today)
        })

    # spara debug-logg
    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    debug_out = os.path.join(PathsConfig.RESULTS_DIR, "debug_ranking.csv")
    pd.DataFrame(daily_logs).to_csv(debug_out, index=False)
    print(f"✅ Sparade debug-logg -> {debug_out}")

if __name__ == "__main__":
    debug_backtest_ranking()
