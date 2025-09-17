# backtest_binary.py
import os
import math
import joblib
import pandas as pd
import numpy as np
from config import PathsConfig, BacktestConfig
from utils import calculate_brokerage_fee, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

def _get_feature_cols(df):
    return [c for c in df.columns if c not in ['date', 'ticker',
                                               'target_regression',
                                               'target_binary',
                                               'target_rank']]

def simulate_engine(df, buy_signals_df, initial_capital, brokerage_fixed_fee, brokerage_percentage, trade_allocation, stop_loss_pct):
    """ 
    Snabb version av backtest-simulatorn. 
    Använder ticker->DataFrame dict för O(1)-uppslag istället för dyra filtreringar.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    all_dates = sorted(df['date'].unique())
    
    # Förbered lookup: {ticker: df_ticker} med date som index
    ticker_dfs = {t: g.set_index('date').sort_index() for t, g in df.groupby('ticker')}
    
    # Buy signal map (date,ticker) -> signal  
    buy_map = buy_signals_df.set_index(['date', 'ticker'])['signal'].to_dict() if not buy_signals_df.empty else {}
    
    trade_log = []
    positions = {}
    cash = initial_capital
    daily_values = []
    
    for date in all_dates:
        # --- Sälj först ---
        for ticker, pos in list(positions.items()):
            tdf = ticker_dfs[ticker]
            if date not in tdf.index:
                continue
                
            # FIX: Hantera Series vs scalar
            price_data = tdf.loc[date, 'adj_close']
            if hasattr(price_data, 'iloc'):
                # Det är en Series, ta första värdet
                price = float(price_data.iloc[0])
            else:
                # Det är redan en scalar
                price = float(price_data)
                
            sell_reason = None
            if price <= pos['purchase_price'] * (1 - stop_loss_pct):
                sell_reason = 'STOP-LOSS'
            if buy_map.get((date, ticker)) == -1:
                sell_reason = 'SELL-SIGNAL'
                
            if sell_reason:
                sale_value = pos['shares'] * price
                fee = calculate_brokerage_fee(sale_value, brokerage_fixed_fee, brokerage_percentage)
                cash += sale_value - fee
                trade_log.append({
                    'date': date,
                    'action': 'SELL', 
                    'ticker': ticker,
                    'price': price,
                    'shares': pos['shares'],
                    'fee': fee,
                    'cash_after': cash,
                    'reason': sell_reason
                })
                positions.pop(ticker)
        
        # --- Köp ---
        buy_candidates = []
        for ticker, tdf in ticker_dfs.items():
            if (date, ticker) in buy_map and buy_map[(date, ticker)] == 1 and ticker not in positions:
                if date in tdf.index:
                    # FIX: Hantera Series vs scalar  
                    price_data = tdf.loc[date, 'adj_close']
                    if hasattr(price_data, 'iloc'):
                        price = float(price_data.iloc[0])
                    else:
                        price = float(price_data)
                    buy_candidates.append({'ticker': ticker, 'price': price})
        
        if buy_candidates and cash > brokerage_fixed_fee:
            capital_to_use = cash * trade_allocation
            capital_per_trade = capital_to_use / len(buy_candidates)
            
            for cand in buy_candidates:
                ticker, price = cand['ticker'], cand['price']
                if price <= 0:
                    continue
                    
                shares = math.floor(capital_per_trade / price)
                if shares <= 0:
                    continue
                    
                buy_cost = shares * price
                fee = calculate_brokerage_fee(buy_cost, brokerage_fixed_fee, brokerage_percentage)
                total_cost = buy_cost + fee
                
                if total_cost <= cash:
                    cash -= total_cost
                    positions[ticker] = {'shares': shares, 'purchase_price': price}
                    trade_log.append({
                        'date': date,
                        'action': 'BUY',
                        'ticker': ticker,
                        'price': price,
                        'shares': shares,
                        'fee': fee,
                        'cash_after': cash,
                        'reason': 'PREDICTION'
                    })
        
        # --- Portföljvärde ---
        port_value = cash
        for t, pos in positions.items():
            tdf = ticker_dfs[t]
            # hitta senaste pris fram till date
            idx = tdf.index.searchsorted(date, side="right") - 1
            if idx >= 0:
                last_price_data = tdf.iloc[idx]['adj_close']
                if hasattr(last_price_data, 'iloc'):
                    last_price = float(last_price_data.iloc[0])
                else:
                    last_price = float(last_price_data)
                port_value += pos['shares'] * last_price
        
        daily_values.append({'date': date, 'portfolio_value': port_value})
    
    trades_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_values)
    return trades_df, daily_df


def backtest_binary():
    print("\n--- Backtest Binary vs Index ---")
    model_path = os.path.join(PathsConfig.MODELS_DIR, "model_binary.pkl")
    if not os.path.exists(model_path):
        print("Ingen binary-modell hittades. Kör train_models först.")
        return

    df = pd.read_parquet(os.path.join(PathsConfig.TARGETS_DIR, "stocks_with_targets.parquet"))
    model = joblib.load(model_path)
    features = _get_feature_cols(df)

    X = df[features].fillna(0)
    # predicted probability for positive class
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['pred_prob'] = probs

    threshold = getattr(BacktestConfig, 'BINARY_THRESHOLD', 0.5)
    df['signal'] = df['pred_prob'].apply(lambda p: 1 if p > threshold else 0)

    buy_signals_df = df[['date','ticker','signal']].copy()

    trades_df, daily_df = simulate_engine(df, buy_signals_df, BacktestConfig.INITIAL_CAPITAL,
                                                             BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE,
                                                             BacktestConfig.TRADE_ALLOCATION, BacktestConfig.STOP_LOSS_PCT)

    if daily_df.empty:
        print("Inga dagvärden genererades.")
        return

    daily_df.set_index('date', inplace=True)
    final_value = float(daily_df['portfolio_value'].iloc[-1])
    total_profit = final_value - BacktestConfig.INITIAL_CAPITAL
    total_fees = trades_df['fee'].sum() if not trades_df.empty else 0.0
    total_trades = len(trades_df)
    daily_returns = daily_df['portfolio_value'].pct_change().dropna()
    sharpe = calculate_sharpe_ratio(daily_returns) if len(daily_returns)>0 else 0.0
    sortino = calculate_sortino_ratio(daily_returns) if len(daily_returns)>0 else 0.0
    maxdd = calculate_max_drawdown(daily_df['portfolio_value'])

    os.makedirs(PathsConfig.RESULTS_DIR, exist_ok=True)
    trades_out = os.path.join(PathsConfig.RESULTS_DIR, "binary_trades.csv")
    daily_out = os.path.join(PathsConfig.RESULTS_DIR, "binary_daily.csv")
    trades_df.to_csv(trades_out, index=False)
    daily_df.to_csv(daily_out)

    print("\n--- Binary backtest summary ---")
    print(f"Slutkapital: {final_value:,.2f} kr")
    print(f"Total vinst: {total_profit:,.2f} kr")
    print(f"Total courtageavgift: {total_fees:,.2f} kr")
    print(f"Totalt antal transaktioner: {total_trades}")
    print(f"Sharpe (år): {sharpe:.2f}, Sortino (år): {sortino:.2f}, MaxDD: {maxdd:.2%}")
    print(f"Sparade trades -> {trades_out}, daglig portfölj -> {daily_out}")
