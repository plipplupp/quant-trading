import os
import math
import pandas as pd
import sqlite3
from config import DatabaseConfig, PathsConfig, BacktestConfig, DataConfig
from utils import calculate_brokerage_fee, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown

def run_omx_benchmark():
    """
    Beräknar och rapporterar avkastningen för att köpa och hålla OMXS30-index.
    """
    print("\n--- Kör Benchmark: Köp & Håll OMXS30 (Marknadsviktat större bolag har större vikt än mindre) ---")
    
    conn = sqlite3.connect(DatabaseConfig.DB_NAME)
    # Läs in all rådata för OMXS30
    df_omx = pd.read_sql("SELECT date, adj_close FROM stocks_raw WHERE ticker = 'OMXS30' ORDER BY date", conn)
    conn.close()

    if df_omx.empty:
        print("Fel: Ingen data för OMXS30 hittades. Kör data_pipeline först.")
        return

    df_omx['date'] = pd.to_datetime(df_omx['date'])
    
    # Hitta start- och slutdatum från hela din datauppsättning för en rättvis jämförelse
    start_date = pd.to_datetime(DataConfig.START_DATE)
    end_date = pd.to_datetime(DataConfig.END_DATE)

    # Filtrera OMX-datan till den relevanta tidsperioden
    df_omx = df_omx[(df_omx['date'] >= start_date) & (df_omx['date'] <= end_date)].copy()
    if df_omx.empty:
        print("Ingen OMX-data för den specificerade tidsperioden.")
        return

    initial_price = df_omx['adj_close'].iloc[0]
    final_price = df_omx['adj_close'].iloc[-1]
    
    # Normalisera värdeutvecklingen
    df_omx['portfolio_value'] = BacktestConfig.INITIAL_CAPITAL * (df_omx['adj_close'] / initial_price)
    
    final_value = df_omx['portfolio_value'].iloc[-1]
    total_profit = final_value - BacktestConfig.INITIAL_CAPITAL
    
    print(f"Startdatum: {df_omx['date'].iloc[0].date()}, Startvärde: {BacktestConfig.INITIAL_CAPITAL:,.0f} kr")
    print(f"Slutdatum: {df_omx['date'].iloc[-1].date()}, Slutvärde: {final_value:,.0f} kr")
    print(f"Total vinst/förlust: {total_profit:,.0f} kr ({total_profit/BacktestConfig.INITIAL_CAPITAL:.2%})")

    # Spara daglig historik
    out_path = os.path.join(PathsConfig.RESULTS_DIR, "benchmark_omx_daily.csv")
    df_omx[['date', 'portfolio_value']].to_csv(out_path, index=False)
    print(f"Daglig historik sparad till: {out_path}")

def run_equal_weight_benchmark():
    """
    Simulerar en "köp och behåll"-strategi för en lika viktad portfölj
    av alla tickers i tickers.txt, inklusive courtage.
    """
    print("\n--- Kör Benchmark: Lika Viktad Köp & Håll Portfölj ---")
    
    conn = sqlite3.connect(DatabaseConfig.DB_NAME)
    df_all = pd.read_sql("SELECT date, ticker, adj_close FROM stocks_prepared ORDER BY date", conn)
    conn.close()

    if df_all.empty:
        print("Fel: Ingen data hittades i 'stocks_prepared'. Kör data_pipeline först.")
        return
        
    df_all['date'] = pd.to_datetime(df_all['date'])
    tickers = DataConfig.TICKERS
    
    # Bestäm den gemensamma start- och slutpunkten för alla aktier
    start_date = df_all['date'].min()
    end_date = df_all['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    initial_capital = float(BacktestConfig.INITIAL_CAPITAL)
    positions = {} # Format: {ticker: {'shares': int, 'invested': float}}
    
    # --- KÖP-LOGIK ---
    # Ta reda på vilka aktier som faktiskt finns vid startdatumet
    available_tickers_at_start = df_all[df_all['date'] == start_date]['ticker'].unique()
    
    # Om en aktie i tickers.txt inte finns vid start, ignoreras den
    tickers_to_buy = [t for t in tickers if t in available_tickers_at_start]
    print(f"Köper {len(tickers_to_buy)} av {len(tickers)} tillgängliga aktier vid startdatum {start_date.date()}.")

    capital_per_ticker = initial_capital / len(tickers_to_buy)
    cash = initial_capital
    
    for ticker in tickers_to_buy:
        price_info = df_all[(df_all['ticker'] == ticker) & (df_all['date'] == start_date)]
        if not price_info.empty:
            price = price_info['adj_close'].iloc[0]
            if price > 0:
                shares_to_buy = math.floor(capital_per_ticker / price)
                buy_cost = shares_to_buy * price
                fee = calculate_brokerage_fee(buy_cost, BacktestConfig.BROKERAGE_FIXED_FEE, BacktestConfig.BROKERAGE_PERCENTAGE)
                
                if shares_to_buy > 0 and (buy_cost + fee) <= capital_per_ticker:
                    positions[ticker] = {'shares': shares_to_buy, 'invested': buy_cost + fee}
                    cash -= (buy_cost + fee)

    # --- DAGLIG VÄRDERING ---
    portfolio_history = []
    
    # Pivotera data för enkel lookup
    price_pivot = df_all.pivot(index='date', columns='ticker', values='adj_close').reindex(all_dates).ffill()

    for date in all_dates:
        portfolio_value = cash
        for ticker, pos in positions.items():
            if ticker in price_pivot.columns:
                last_price = price_pivot.loc[date, ticker]
                portfolio_value += pos['shares'] * last_price
        portfolio_history.append({'date': date, 'portfolio_value': portfolio_value})

    # --- SUMMERING ---
    daily_df = pd.DataFrame(portfolio_history)
    final_value = daily_df['portfolio_value'].iloc[-1]
    total_profit = final_value - initial_capital
    
    print(f"Startdatum: {start_date.date()}, Startvärde: {initial_capital:,.0f} kr")
    print(f"Slutdatum: {end_date.date()}, Slutvärde: {final_value:,.0f} kr")
    print(f"Total vinst/förlust: {total_profit:,.0f} kr ({total_profit/initial_capital:.2%})")

    out_path = os.path.join(PathsConfig.RESULTS_DIR, "benchmark_equal_weight_daily.csv")
    daily_df.to_csv(out_path, index=False)
    print(f"Daglig historik sparad till: {out_path}")