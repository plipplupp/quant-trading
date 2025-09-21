# debug_backtest_random_universe.py
import os
import math
import random
import joblib
import pandas as pd
import numpy as np
from config import PathsConfig, BacktestConfig, TargetConfig
from utils import calculate_brokerage_fee, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

# snabbkoll av konfig
print("=== BacktestConfig (snabbkoll) ===")
attrs = ['INITIAL_CAPITAL','TRADE_ALLOCATION','BROKERAGE_FIXED_FEE','BROKERAGE_PERCENTAGE','STOP_LOSS_PCT']
for a in attrs:
    print(a, "=", getattr(BacktestConfig, a, None))
print("TargetConfig RANK_REBALANCE_DAYS:", getattr(TargetConfig, 'RANK_REBALANCE_DAYS', None))


# --- Konfig för debugrun ---
RANDOM_SEED = 42
SAMPLE_N = 10  # antal slump-tickers per rebalansering
SAVE_PREFIX = "debug_random_"  # prefix för output-filer

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _extract_price_safe(row, preferred=('adj_close', 'close', 'open')):
    for p in preferred:
        if p in row.index:
            val = row[p]
            if isinstance(val, pd.Series):
                val = val.dropna()
                if not val.empty:
                    val = val.iloc[0]
                else:
                    continue
            try:
                val = float(val)
            except Exception:
                continue
            if pd.notnull(val) and val > 0:
                return val
    return None


def debug_backtest_random_universe():
    print("\n--- DEBUG Backtest: Random Universe ---")

    # 1) Ladda data
    data_path = os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet")
    if not os.path.exists(data_path):
        print(f"Fel: Kan inte hitta {data_path}")
        return

    df_all = pd.read_parquet(data_path)
    df_all['date'] = pd.to_datetime(df_all['date']).dt.normalize()

    # enkel datakvalitetskontroll på adj_close
    print("Data: antal rader:", len(df_all))
    desc = df_all['adj_close'].describe()
    print("Adj_close summary:", desc.to_dict())

    # välj universe (alla tickers) och gör enkel sanity-check
    universe = sorted(df_all['ticker'].unique().tolist())
    print(f"Totalt tickers i data: {len(universe)}")

    if len(universe) < SAMPLE_N:
        print(f"För få tickers ({len(universe)}) för sample_n={SAMPLE_N}. Avslutar.")
        return

    # 2) ladda (eller hoppa över) modellen — här behöver vi den inte, men håller möjlighet
    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_ranking.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("Modellen laddad — men används ej i denna debugkörning.")
        except Exception as e:
            print("Kan ej ladda modell (ok). Fel:", e)

    # 3) filtera dataframe för bara de tickers vi behöver (hela universe används för sampling)
    df = df_all.copy()
    df.dropna(subset=['ticker', 'date', 'adj_close'], inplace=True)

    # Prebuild per-ticker dataframes indexerade på date
    ticker_dfs = {t: g.set_index('date').sort_index() for t, g in df.groupby('ticker')}
    all_dates = sorted(df['date'].unique())

    # init
    rebalance_freq = getattr(TargetConfig, 'RANK_REBALANCE_DAYS', 5)
    cash = float(BacktestConfig.INITIAL_CAPITAL)
    positions = {}  # {ticker: {'shares': int, 'purchase_price': float}}
    pending_signals = []  # {'action','ticker','exec_date','reason'}
    trade_log = []
    daily_logs = []

    # hjälp för pending-signals utan duplicat
    def _add_pending(action, ticker, exec_date, reason):
        for s in pending_signals:
            if s['action'] == action and s['ticker'] == ticker and s['exec_date'] == exec_date:
                return
        pending_signals.append({'action': action, 'ticker': ticker, 'exec_date': exec_date, 'reason': reason})

    # keep track för debug-statistik
    max_single_position_value = 0
    max_shares_any = 0

    # MAIN LOOP över dagar
    for i, today in enumerate(all_dates):
        # 1) mark-to-market portföljvärde (använd dagens adj_close eller senaste tidigare)
        port_value = cash
        for t, pos in positions.items():
            tdf = ticker_dfs.get(t)
            last_price = pos['purchase_price']
            if tdf is not None:
                if today in tdf.index:
                    # säkert extrahera
                    val = tdf.loc[today, 'adj_close']
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    last_price = float(val)
                else:
                    idx = tdf.index.searchsorted(today, side='right') - 1
                    if idx >= 0:
                        val = tdf.iloc[idx]['adj_close']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0]
                        last_price = float(val)
            pos_value = pos['shares'] * last_price
            port_value += pos_value
            max_single_position_value = max(max_single_position_value, pos_value)
            max_shares_any = max(max_shares_any, pos['shares'])

        # 2) exekvera pending signals idag (SELL på open, BUY på close)
        todays_sells = [s for s in pending_signals if s['exec_date'] == today and s['action'] == 'SELL']
        todays_buys = [s for s in pending_signals if s['exec_date'] == today and s['action'] == 'BUY']
        # rensa executed
        pending_signals = [s for s in pending_signals if s['exec_date'] != today]

        # SELL -> OPEN
        for sig in todays_sells:
            t = sig['ticker']
            if t not in positions:
                continue
            tdf = ticker_dfs.get(t)
            if tdf is None or today not in tdf.index:
                continue
            row = tdf.loc[today]
            sell_price = _extract_price_safe(row, preferred=('open', 'adj_close', 'close'))
            if sell_price is None:
                continue
            shares = positions[t]['shares']
            sale_value = shares * sell_price
            fee = calculate_brokerage_fee(sale_value, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
            cash += sale_value - fee
            trade_log.append({'date': today, 'action': 'SELL', 'ticker': t, 'price': sell_price, 'shares': shares, 'fee': fee, 'cash_after': cash, 'reason': sig.get('reason', '')})
            del positions[t]

        # BUY -> CLOSE (använd cash * TRADE_ALLOCATION delat jämt mellan buys)
        if todays_buys:
            capital_to_use = float(cash) * float(BacktestConfig.TRADE_ALLOCATION)
            n_buys = len(todays_buys)
            capital_per = capital_to_use / max(n_buys, 1)

            for sig in sorted(todays_buys, key=lambda x: x['ticker']):
                t = sig['ticker']
                if t in positions:
                    continue
                tdf = ticker_dfs.get(t)
                if tdf is None or today not in tdf.index:
                    continue
                row = tdf.loc[today]
                buy_price = _extract_price_safe(row, preferred=('adj_close', 'close', 'open'))
                if buy_price is None:
                    continue

                shares = math.floor(capital_per / buy_price)
                # thresholds
                SHARES_ALERT = 1_000_000   # tröskel som du redan såg bryts
                CASH_ALERT = 1e9           # om cash > 1 miljard — indikativt fel

                if shares > SHARES_ALERT or cash > CASH_ALERT:
                    debug_path = os.path.join(PathsConfig.RESULTS_DIR, "debug_large_trade_snapshot.txt")
                    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write(f"ALERT large trade at i={i}, date={today}\n")
                        f.write(f"cash (före köp) = {cash}\n")
                        f.write(f"capital_to_use = {capital_to_use}\n")
                        f.write(f"capital_per = {capital_per}\n")
                        f.write(f"buy_price = {buy_price}\n")
                        f.write(f"shares_calculated = {shares}\n")
                        f.write(f"len(positions) = {len(positions)}\n")
                        f.write("Current positions snapshot (top 20):\n")
                        for k,v in list(positions.items())[:20]:
                            f.write(f"  {k}: shares={v['shares']}, purchase_price={v['purchase_price']}\n")
                        f.write("\nLast 30 trades (if any):\n")
                        try:
                            last_trades = trade_log[-30:]
                            for tr in last_trades:
                                f.write(str(tr) + "\n")
                        except Exception as e:
                            f.write("Could not dump trade_log: " + str(e))
                        f.write("\nSample rows from df for this ticker around date:\n")
                        try:
                            if t in ticker_dfs and today in ticker_dfs[t].index:
                                f.write(str(ticker_dfs[t].loc[today- pd.Timedelta(days=7): today + pd.Timedelta(days=7)].head(20)))
                        except Exception as e:
                            f.write("Could not extract df sample: " + str(e))
                    print("!!! ALERT: Large trade snapshot written to", debug_path)
                    # OPTIONAL: break the run (uncomment if you want immediate stop)
                    # raise SystemExit("Stopped due to large shares detected (see debug snapshot)")

                while shares > 0:
                    buy_cost = shares * buy_price
                    fee = calculate_brokerage_fee(buy_cost, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
                    total_cost = buy_cost + fee
                    if total_cost <= cash:
                        break
                    shares -= 1
                if shares <= 0:
                    continue
                buy_cost = shares * buy_price
                fee = calculate_brokerage_fee(buy_cost, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
                cash -= (buy_cost + fee)
                positions[t] = {'shares': shares, 'purchase_price': buy_price}
                trade_log.append({'date': today, 'action': 'BUY', 'ticker': t, 'price': buy_price, 'shares': shares, 'fee': fee, 'cash_after': cash, 'reason': sig.get('reason', '')})
                max_single_position_value = max(max_single_position_value, shares * buy_price)
                max_shares_any = max(max_shares_any, shares)

        # 3) schedule next-day trades by picking random SAMPLE_N tickers on rebal-days
        if i % rebalance_freq == 0:
            # välj slumpmässigt SAMPLE_N tickers från universe
            selected = random.sample(universe, SAMPLE_N)
            selected_set = set(selected)
            current_set = set(positions.keys())
            to_sell = current_set - selected_set
            to_buy = selected_set - current_set
            if i + 1 < len(all_dates):
                exec_date = all_dates[i + 1]
                for t in to_sell:
                    _add_pending('SELL', t, exec_date, 'REBALANCE_OUT_random')
                for t in to_buy:
                    _add_pending('BUY', t, exec_date, 'REBALANCE_IN_random')

        # 4) simple stop-loss (use today's adj_close vs purchase_price)
        for t, pos in list(positions.items()):
            tdf = ticker_dfs.get(t)
            if tdf is None or today not in tdf.index:
                continue
            val = tdf.loc[today, 'adj_close']
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            current_close = float(val)
            if current_close <= pos['purchase_price'] * (1 - float(BacktestConfig.STOP_LOSS_PCT)):
                if i + 1 < len(all_dates):
                    exec_date = all_dates[i + 1]
                    _add_pending('SELL', t, exec_date, 'STOP_LOSS_random')

        # 5) Logga dagligt värde
        daily_logs.append({'date': today, 'portfolio_value': int(round(port_value)), 'cash': int(round(cash)), 'positions_count': len(positions)})

        # snabba varningar
        if cash < -1e-6:
            print(f"VARNING: cash negativt på {today}: {cash}")

    # efter loop: skriv ut debug-info & filer
    trades_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_logs).set_index('date')

    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    trades_out = os.path.join(PathsConfig.RESULTS_DIR, SAVE_PREFIX + "trades.csv")
    daily_out = os.path.join(PathsConfig.RESULTS_DIR, SAVE_PREFIX + "daily.csv")
    sanity_out = os.path.join(PathsConfig.RESULTS_DIR, SAVE_PREFIX + "sanity_checks.txt")

    if not trades_df.empty:
        trades_df.to_csv(trades_out, index=False)
    daily_df.to_csv(daily_out)

    # sanity checks
    with open(sanity_out, "w") as f:
        f.write(f"Total trades: {len(trade_log)}\n")
        f.write(f"Final portfolio value: {daily_df['portfolio_value'].iloc[-1] if not daily_df.empty else 'NA'}\n")
        f.write(f"Max single position value seen: {max_single_position_value}\n")
        f.write(f"Max shares for any ticker: {max_shares_any}\n")
        # kontrollera om några trades använder pris > some threshold eller shares huge
        large_price_trades = trades_df[trades_df['price'] > 1_000_000] if not trades_df.empty else pd.DataFrame()
        if not large_price_trades.empty:
            f.write("Trades with price > 1,000,000 found:\n")
            f.write(large_price_trades.to_csv(index=False))
        high_shares = trades_df[trades_df['shares'] > 1_000_000] if not trades_df.empty else pd.DataFrame()
        if not high_shares.empty:
            f.write("Trades with shares > 1,000,000 found:\n")
            f.write(high_shares.to_csv(index=False))
        # kontrollera att alla trade-prices matchar adj_close/open for ticker/date (snabb heuristik)
        mismatches = []
        for _, row in trades_df.iterrows():
            t = row['ticker']; d = pd.to_datetime(row['date']).normalize()
            tdf = ticker_dfs.get(t)
            if tdf is None:
                mismatches.append((row.to_dict(), "no_tdf"))
                continue
            if d not in tdf.index:
                mismatches.append((row.to_dict(), "date_missing"))
                continue
            # allow small rounding diffs
            row_price = float(row['price'])
            expected_vals = []
            for col in ('open', 'close', 'adj_close'):
                if col in tdf.columns:
                    v = tdf.loc[d, col]
                    if isinstance(v, pd.Series):
                        v = v.iloc[0]
                    expected_vals.append(float(v))
            # compare to any expected within 0.5% tolerance
            ok = any(abs(row_price - ev) / max(1e-9, ev) < 0.005 for ev in expected_vals)
            if not ok:
                mismatches.append((row.to_dict(), expected_vals))
        f.write(f"Number of trades with price mismatch vs stored price columns: {len(mismatches)}\n")
        if mismatches:
            f.write("Sample mismatches (up to 10):\n")
            for m in mismatches[:10]:
                f.write(str(m) + "\n")

    print("\nDEBUG run finished.")
    print(f"Trades -> {trades_out}")
    print(f"Daily -> {daily_out}")
    print(f"Sanity -> {sanity_out}")
    # print quick summary
    if not daily_df.empty:
        final_val = float(daily_df['portfolio_value'].iloc[-1])
        print(f"Final portfolio value: {final_val:,.2f} (cash {int(round(cash))}, trades {len(trade_log)})")


if __name__ == "__main__":
    debug_backtest_random_universe()
