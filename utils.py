import math
import pandas as pd
import numpy as np

# Simuleringsmotor för backtesting av tradingstrategierna regression och binär klassificering
def simulate_engine(df, buy_signals_df, initial_capital, brokerage_fixed_fee, brokerage_percentage, trade_allocation, stop_loss_pct):
    """
    Backtest-simulator med D+1 exekveringslogik.
    - Använder handlingsbara priser: SÄLJ på OPEN, KÖP på CLOSE.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    all_dates = sorted(df['date'].unique())
    ticker_dfs = {t: g.set_index('date').sort_index() for t, g in df.groupby('ticker')}
    signal_map = buy_signals_df.set_index(['date', 'ticker'])['signal'].to_dict() if not buy_signals_df.empty else {}

    cash = float(initial_capital)
    positions = {}
    pending_signals = []
    trade_log = []
    daily_logs = []

    def _add_pending(action, ticker, exec_date, reason):
        for s in pending_signals:
            if s['action'] == action and s['ticker'] == ticker and s['exec_date'] == exec_date:
                return
        pending_signals.append({'action': action, 'ticker': ticker, 'exec_date': exec_date, 'reason': reason})

    for i, today in enumerate(all_dates):
        next_day = all_dates[i + 1] if i + 1 < len(all_dates) else None

        # 1. EXEKVERA DAGENS AFFÄRER
        todays_sells = [s for s in pending_signals if s['exec_date'] == today and s['action'] == 'SELL']
        todays_buys = [s for s in pending_signals if s['exec_date'] == today and s['action'] == 'BUY']
        pending_signals = [s for s in pending_signals if s['exec_date'] != today]

        # SÄLJ FÖRST - på dagens ÖPPNINGSKURS (OPEN)
        for sig in todays_sells:
            ticker = sig['ticker']
            if ticker not in positions: continue
            tdf = ticker_dfs.get(ticker)
            if tdf is None or today not in tdf.index: continue
            row = tdf.loc[today]
            sell_price = _extract_price_safe(row, preferred=('open', 'close')) # Fallback till 'close' om 'open' saknas
            
            if sell_price is None: continue

            shares = positions[ticker]['shares']
            sale_value = shares * sell_price
            fee = calculate_brokerage_fee(sale_value, brokerage_fixed_fee, brokerage_percentage)
            cash += sale_value - fee
            
            trade_log.append({
                'date': today, 'action': 'SELL', 'ticker': ticker,
                'price': sell_price, 'shares': shares, 'fee': fee,
                'cash_after': cash, 'reason': sig.get('reason', 'N/A')
            })
            del positions[ticker]

        # KÖP SEDAN - på dagens STÄNGNINGSKURS (CLOSE)
        if todays_buys:
            capital_to_use = float(cash) * float(trade_allocation)
            capital_per_trade = capital_to_use / max(1, len(todays_buys))

            for sig in sorted(todays_buys, key=lambda x: x['ticker']):
                ticker = sig['ticker']
                if ticker in positions: continue
                tdf = ticker_dfs.get(ticker)
                if tdf is None or today not in tdf.index: continue
                row = tdf.loc[today]
                buy_price = _extract_price_safe(row, preferred=('close', 'open')) # Fallback till 'open' om 'close' saknas
                
                if buy_price is None: continue

                shares = math.floor(capital_per_trade / buy_price)
                if shares <= 0: continue
                
                buy_cost = shares * buy_price
                fee = calculate_brokerage_fee(buy_cost, brokerage_fixed_fee, brokerage_percentage)
                if buy_cost + fee > cash:
                    shares = math.floor((cash - fee) / buy_price)
                
                if shares <= 0: continue
                
                total_cost = (shares * buy_price) + calculate_brokerage_fee(shares * buy_price, brokerage_fixed_fee, brokerage_percentage)
                cash -= total_cost
                positions[ticker] = {'shares': shares, 'purchase_price': buy_price}

                trade_log.append({
                    'date': today, 'action': 'BUY', 'ticker': ticker,
                    'price': buy_price, 'shares': shares, 'fee': fee,
                    'cash_after': cash, 'reason': sig.get('reason', 'N/A')
                })

        # 2. SKAPA NYA SIGNALER FÖR MORGONDAGEN
        if next_day:
            # A. Stop-loss signaler
            for ticker, pos in list(positions.items()):
                tdf = ticker_dfs.get(ticker)
                if tdf is not None and today in tdf.index:
                    # Viktigt: Stop-loss kan fortfarande baseras på adj_close för att mäta "sann" förlust
                    current_close = float(tdf.loc[today, 'adj_close'])
                    if current_close <= pos['purchase_price'] * (1 - stop_loss_pct):
                        _add_pending('SELL', ticker, next_day, 'STOP_LOSS')

            # B. Modellbaserade signaler
            for (sig_date, ticker), signal_value in signal_map.items():
                if sig_date == today:
                    if signal_value == 1 and ticker not in positions:
                        _add_pending('BUY', ticker, next_day, 'BUY_SIGNAL')
                    elif signal_value == -1 and ticker in positions:
                        _add_pending('SELL', ticker, next_day, 'SELL_SIGNAL')
        
        # 3. LOGGA DAGLIGT PORTFÖLJVÄRDE
        portfolio_value = cash
        for ticker, pos in positions.items():
            tdf = ticker_dfs.get(ticker)
            
            # Sätt ett fallback-pris (inköpspriset) om vi inte hittar dagens kurs
            last_known_price = pos['purchase_price'] 
            
            if tdf is not None:
                # Försök hitta dagens data
                if today in tdf.index:
                    # Hämta dagens adj_close för en korrekt värdering.
                    last_known_price = tdf.loc[today, 'adj_close']
                else:
                    # Om dagens data saknas (t.ex. helgdag), ta den senast kända.
                    # Din söklogik med searchsorted hanterade detta bra.
                    idx = tdf.index.searchsorted(today, side='right') - 1
                    if idx >= 0:
                        last_known_price = tdf.iloc[idx]['adj_close']
                        
            portfolio_value += pos['shares'] * float(last_known_price)
                    
        daily_logs.append({'date': today, 'portfolio_value': portfolio_value})
    
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    daily_df = pd.DataFrame(daily_logs) if daily_logs else pd.DataFrame()

    return trades_df, daily_df

def calculate_brokerage_fee(transaction_cost, fixed_fee, percentage_fee):
    """
    Beräknar courtageavgiften baserat på transaktionskostnaden.
    """
    calculated_percentage_fee = transaction_cost * percentage_fee
    return max(fixed_fee, calculated_percentage_fee)

def _extract_price_safe(row, preferred=('adj_close', 'close', 'open')):
    for p in preferred:
        if p in row.index:  # radens kolumner
            val = row[p]
            if isinstance(val, pd.Series):  # ibland en serie
                val = val.iloc[0]
            if pd.notnull(val) and val > 0:
                return float(val)
    return None

def create_future_label(df, days, threshold):
    """
    Skapar målvariabeln (label) baserat på framtida prisrörelse.
    """
    if df.empty or 'adj_close' not in df.columns:
        return df.copy()
        
    df_copy = df.copy()
    price_future = df_copy['adj_close'].shift(-days)
    price_change = (price_future - df_copy['adj_close']) / df_copy['adj_close']
    
    df_copy['future_label'] = 'Behåll'
    df_copy.loc[price_change >= threshold, 'future_label'] = 'Köp'
    df_copy.loc[price_change <= -threshold, 'future_label'] = 'Sälj'
    
    df_copy.iloc[-days:, df_copy.columns.get_loc('future_label')] = np.nan
    
    return df_copy


def calculate_atr(high, low, close, window=14):
    """Beräknar Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window).mean()
    return atr

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Beräknar den årliga Sharpekvoten."""
    excess_returns = returns - risk_free_rate / 252
    # Årlig Sharpekvot: sqrt(252) * medelavkastning / std på avkastning
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Beräknar den årliga Sortinokvoten."""
    excess_returns = returns - risk_free_rate / 252
    # Beräkna standardavvikelsen för endast negativ avkastning (nedåtrisk)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf # Oändlig om ingen nedåtrisk finns
        
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
    return sortino_ratio if np.isfinite(sortino_ratio) else 0.0

def calculate_max_drawdown(prices):
    """Beräknar den maximala nedgången från en topp (Max Drawdown)."""
    # Vi beräknar den löpande maximala toppen
    running_max = prices.expanding().max()
    # Beräkna nedgången från den löpande toppen
    drawdown = (prices - running_max) / running_max
    return drawdown.min()

def clean_and_filter_data(df, price_col='adj_close', min_price=10.0, max_daily_return=0.50):
    """
    Rensar bort aktier (tickers) som inte uppfyller datakrav:
    - För stora dagliga rörelser
    - För ofta under minsta prisnivå

    Returns:
        df (pd.DataFrame): Rensad DataFrame
        removed_tickers (list): Lista med borttagna tickers
    """
    print("\n--- Datakvalitetskontroll & Sanering ---")

    tickers_before = df['ticker'].nunique()
    removed_tickers = []

    # Beräkna daglig avkastning
    df['daily_return'] = df.groupby('ticker')[price_col].pct_change()

    # 1. Ta bort tickers med extrema rörelser
    extreme_movers = df[df['daily_return'].abs() > max_daily_return]['ticker'].unique().tolist()
    if extreme_movers:
        print(f"⚠️ Tar bort {len(extreme_movers)} tickers med extrema dagliga rörelser (> {max_daily_return:.0%}): {', '.join(extreme_movers)}")
        df = df[~df['ticker'].isin(extreme_movers)]
        removed_tickers.extend(extreme_movers)

    # 2. Ta bort penny stocks
    low_price_tickers = df[df[price_col] < min_price]['ticker'].value_counts()
    penny_stocks = low_price_tickers[low_price_tickers > 10].index.tolist()
    if penny_stocks:
        print(f"⚠️ Tar bort {len(penny_stocks)} tickers som ofta handlas under {min_price} kr: {', '.join(penny_stocks)}")
        df = df[~df['ticker'].isin(penny_stocks)]
        removed_tickers.extend(penny_stocks)

    tickers_after = df['ticker'].nunique()
    removed_count = tickers_before - tickers_after

    print(f"✅ Datasanering slutförd. (borttagna tickers: {removed_count})")
    
    # Städa upp
    df = df.drop(columns=['daily_return'])
    return df, removed_tickers

