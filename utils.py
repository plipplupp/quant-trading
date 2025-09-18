import pandas as pd
import numpy as np

def calculate_brokerage_fee(transaction_cost, fixed_fee, percentage_fee):
    """
    Beräknar courtageavgiften baserat på transaktionskostnaden.
    """
    calculated_percentage_fee = transaction_cost * percentage_fee
    return max(fixed_fee, calculated_percentage_fee)

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

