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