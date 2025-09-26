# backtest_ranking.py  (ersätt din gamla backtest_ranking med denna)
import os
import math
import joblib
import pandas as pd
from config import PathsConfig, BacktestConfig, TargetConfig
from utils import (
    calculate_brokerage_fee,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    _extract_price_safe
)

def _get_feature_cols(df):
    return [c for c in df.columns if c not in [
        'date', 'ticker',
        'target_regression', 'target_binary', 'target_rank'
    ]]

def backtest_ranking():
    """
    Robust D+1 ranking backtest:
      - Signaler genereras baserat på dagens predicted_score.
      - Signaler exekveras nästa handelsdag: SELL på OPEN, BUY på CLOSE.
      - Köpstorlek baseras på cash * TRADE_ALLOCATION och delas mellan dagens buys.
      - Om kostnad inte ryms i cash minskar vi antalet aktier tills det gör det.
    """
    print("\n--- Backtest Ranking (robust D+1) ---")
    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl")
    if not os.path.exists(model_path):
        print("Ingen ranking-modell hittades. Kör train_models först.")
        return

    # --- Ladda data & modell ---
    df = pd.read_parquet(os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet"))
    model = joblib.load(model_path)

    # Kontrollera att modellen är en pipeline med scaler (om möjligt)
    try:
        import sklearn
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            print("OK model is a pipeline and includes scaler.")
        else:
            print("VARNING: modeller saknar scaler i pipeline - kontrollera hur modellen sparades.")
    except Exception as e:
        print("Could not inspect model:", e)


    # Normalisera datum (enbart datum)
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # Förutsätt att nödvändiga pris-kolumner finns
    df.dropna(subset=['ticker','date','adj_close'], inplace=True)

    features = _get_feature_cols(df)
    # Predict — använd samma features som modellen tränades på
    X = df[features].fillna(0)
    df['predicted_score'] = model.predict(X)

    # --- Init ---
    top_n = getattr(TargetConfig, 'RANK_TOP_N', 10)
    top_pct = getattr(TargetConfig, 'RANK_TOP_PCT', None)
    rebalance_freq = getattr(TargetConfig, 'RANK_REBALANCE_DAYS', 5)

    all_dates = sorted(df['date'].unique())
    # snabb åtkomst per ticker (indexerade på date)
    ticker_dfs = {t: g.set_index('date').sort_index() for t, g in df.groupby('ticker')}

    cash = float(BacktestConfig.INITIAL_CAPITAL)
    positions = {}  # {ticker: {'shares': int, 'purchase_price': float}}
    pending_signals = []  # list of {'action','ticker','exec_date','reason'}
    trade_log = []
    daily_logs = []

    # Helper för att undvika dubbletter i pending_signals
    def _add_pending(action, ticker, exec_date, reason):
        for s in pending_signals:
            if s['action']==action and s['ticker']==ticker and s['exec_date']==exec_date:
                return
        pending_signals.append({'action': action, 'ticker': ticker, 'exec_date': exec_date, 'reason': reason})

    # --- Loop över handelsdagar ---
    for i, today in enumerate(all_dates):
        # 1) Börja med att värdera portföljen *med dagens priser* om möjligt:
        portfolio_value = cash
        for t, pos in positions.items():
            tdf = ticker_dfs.get(t)
            last_price = None
            if tdf is not None:
                if today in tdf.index:
                    last_price = float(tdf.loc[today, 'adj_close'])
                else:
                    idx = tdf.index.searchsorted(today, side='right') - 1
                    if idx >= 0:
                        last_price = float(tdf.iloc[idx]['adj_close'])
            if last_price is None:
                last_price = pos['purchase_price']
            portfolio_value += pos['shares'] * last_price

        # 2) Exekvera signals för idag (SELL på OPEN, BUY på CLOSE)
        todays_sells = [s for s in pending_signals if s['exec_date'] == today and s['action']=='SELL']
        todays_buys  = [s for s in pending_signals if s['exec_date'] == today and s['action']=='BUY']
        # rensa de som vi ska exekvera idag
        pending_signals = [s for s in pending_signals if s['exec_date'] != today]

        # SELL först — exekvera på OPEN (eller fallback)
        for sig in todays_sells:
            t = sig['ticker']
            if t not in positions:
                continue
            tdf = ticker_dfs.get(t)
            if tdf is None or today not in tdf.index:
                # ingen prisdata för att sälja idag — hoppa
                continue
            # hämta open först, fallback till close
            row = tdf.loc[today]
            sell_price = _extract_price_safe(row, preferred=('open','close'))
            if sell_price is None or sell_price <= 0:
                continue
            shares = positions[t]['shares']
            sale_value = shares * sell_price
            fee = calculate_brokerage_fee(sale_value, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
            cash += sale_value - fee
            trade_log.append({'date': today, 'action': 'SELL', 'ticker': t,
                              'price': sell_price, 'shares': shares, 'fee': fee,
                              'cash_after': cash, 'reason': sig.get('reason','SELL')})
            del positions[t]

        # BUY på CLOSE — använd cash * TRADE_ALLOCATION för dagens buys
        if todays_buys:
            # bestäm hur mycket kapital vi låter användas idag
            capital_to_use = float(cash) * float(BacktestConfig.TRADE_ALLOCATION)
            # dela jämnt mellan buys (vi justerar shares se nedan vid behov)
            n_buys = len(todays_buys)
            capital_per = capital_to_use / max(n_buys, 1)

            # Viktigt: iterera buys i deterministic ordning (t.ex. sortera ticker) för reproducerbarhet
            for sig in sorted(todays_buys, key=lambda x: x['ticker']):
                t = sig['ticker']
                # hoppa om vi redan äger (kan hända i corner cases)
                if t in positions:
                    continue
                tdf = ticker_dfs.get(t)
                if tdf is None or today not in tdf.index:
                    continue
                row = tdf.loc[today]
                buy_price = _extract_price_safe(row, preferred=('close','open'))
                if buy_price is None or buy_price <= 0:
                    continue

                # initial antal shares utifrån capital_per
                shares = math.floor(capital_per / buy_price)
                # minska tills buy_cost + fee <= cash (så att vi inte går negativ)
                while shares > 0:
                    buy_cost = shares * buy_price
                    fee = calculate_brokerage_fee(buy_cost, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
                    total_cost = buy_cost + fee
                    if total_cost <= cash:
                        break
                    shares -= 1

                if shares <= 0:
                    # kan inte köpa denna ticker idag
                    continue

                # genomför köp
                buy_cost = shares * buy_price
                fee = calculate_brokerage_fee(buy_cost, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
                total_cost = buy_cost + fee
                cash -= total_cost
                positions[t] = {'shares': shares, 'purchase_price': buy_price}
                trade_log.append({'date': today, 'action': 'BUY', 'ticker': t,
                                  'price': buy_price, 'shares': shares, 'fee': fee,
                                  'cash_after': cash, 'reason': sig.get('reason','BUY')})

        # 3) Skapa signaler för morgondagen baserat på dagens data (D -> exec_date = nästa handelsdag)
        # Stop-loss (om dagens close <= purchase_price*(1-STOP_LOSS_PCT) -> SELL next day)
        for t, pos in list(positions.items()):
            tdf = ticker_dfs.get(t)
            if tdf is None or today not in tdf.index:
                continue
            current_close = float(tdf.loc[today, 'close'])
            if current_close <= pos['purchase_price'] * (1 - float(BacktestConfig.STOP_LOSS_PCT)):
                # schedule sell next trading day (om den finns)
                if i + 1 < len(all_dates):
                    exec_date = all_dates[i+1]
                    _add_pending('SELL', t, exec_date, 'STOP_LOSS')

        # Rebalansering: planera SELL/BUY för nästa dag baserat på today's ranking
        if i % rebalance_freq == 0:
            todays_data = df[df['date'] == today]
            if not todays_data.empty:
                if top_pct is not None:
                    cutoff = todays_data['predicted_score'].quantile(1 - top_pct)
                    selected = set(todays_data[todays_data['predicted_score'] >= cutoff]['ticker'].tolist())
                else:
                    selected = set(todays_data.nlargest(top_n, 'predicted_score')['ticker'].tolist())

                current_holding = set(positions.keys())
                to_sell = current_holding - selected
                to_buy  = selected - current_holding

                if i + 1 < len(all_dates):
                    exec_date = all_dates[i+1]
                    for t in to_sell:
                        _add_pending('SELL', t, exec_date, 'REBALANCE_OUT')
                    for t in to_buy:
                        _add_pending('BUY', t, exec_date, 'REBALANCE_IN')

        # 4) Logga dagligt värde (runda till heltal)
        # återberäkna mark-to-market med dagens adj_close om möjligt
        port_value = cash
        for t, pos in positions.items():
            tdf = ticker_dfs.get(t)
            last_price = pos['purchase_price']
            if tdf is not None:
                if today in tdf.index:
                    last_price = float(tdf.loc[today,'adj_close'])
                else:
                    idx = tdf.index.searchsorted(today, side='right') - 1
                    if idx >= 0:
                        last_price = float(tdf.iloc[idx]['adj_close'])
            port_value += pos['shares'] * last_price

        daily_logs.append({'date': today, 'portfolio_value': int(round(port_value)), 'cash': int(round(cash)), 'positions_count': len(positions)})

        # säkerhetskontroller (snabb varning)
        if cash < -1e-6:
            print(f"VARNING: cash negativt på {today}: {cash:.2f}")

    # --- Summera och spara resultat ---
    trades_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_logs).set_index('date')

    final_value = float(daily_df['portfolio_value'].iloc[-1])
    total_profit = final_value - float(BacktestConfig.INITIAL_CAPITAL)
    total_fees = trades_df['fee'].sum() if not trades_df.empty else 0.0
    total_trades = len(trades_df)
    daily_returns = daily_df['portfolio_value'].pct_change().dropna()

    sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0
    sortino = calculate_sortino_ratio(daily_returns) if len(daily_returns) > 0 else 0.0
    maxdd = calculate_max_drawdown(daily_df['portfolio_value'])

    # =========================================================
    # Skapa signal-CSV för senaste datumet
    # =========================================================
    print("\nSkapar dagens rankningssignaler...")
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].copy()

    # Sortera efter predicted_score (högst är bäst)
    latest_df = latest_df.sort_values(by='predicted_score', ascending=False)

    # Välj kolumner att spara
    signals_out_path = os.path.join(PathsConfig.RESULTS_DIR, "ranking_signals_today.csv")
    latest_df[['ticker', 'predicted_score']].to_csv(signals_out_path, index=False)
    # =========================================================
    
    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    trades_out = os.path.join(PathsConfig.RESULTS_DIR, "ranking_trades.csv")
    daily_out = os.path.join(PathsConfig.RESULTS_DIR, "ranking_daily.csv")
    trades_df.to_csv(trades_out, index=False)
    daily_df.to_csv(daily_out)

    print(f"Dagens rankning ({latest_date.date()}): sparad till {signals_out_path}")

    print("\n--- Ranking backtest summary ---")
    print(f"Slutkapital: {final_value:,.2f} kr")
    print(f"Total vinst: {total_profit:,.2f} kr")
    print(f"Total courtageavgift: {total_fees:,.2f} kr")
    print(f"Totalt antal transaktioner: {total_trades}")
    print(f"Sharpe (år): {sharpe:.2f}, Sortino (år): {sortino:.2f}, MaxDD: {maxdd:.2%}")
    print(f"Sparade trades -> {trades_out}, daglig portfölj -> {daily_out}")

if __name__ == "__main__":
    backtest_ranking()
