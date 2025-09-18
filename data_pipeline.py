import yfinance as yf
import sqlite3
import pandas as pd
import numpy as np
from config import DataConfig, DatabaseConfig
from datetime import datetime, timedelta
from utils import clean_and_filter_data

# =============================================================================
# HJÄLPFUNKTIONER FÖR FEATURE-BERÄKNING (Dina avancerade funktioner, inga ändringar här)
# =============================================================================

def _add_moving_averages(df):
    windows = [5, 10, 15, 20, 30, 40, 50, 60, 100, 200]
    for n in windows:
        df[f'sma_{n}'] = df['adj_close'].rolling(window=n, min_periods=1).mean()
        df[f'ema_{n}'] = df['adj_close'].ewm(span=n, adjust=False).mean()
    df['sma_10_20_diff'] = df['sma_10'] - df['sma_20']
    df['sma_50_200_diff'] = df['sma_50'] - df['sma_200']
    df['ema_10_50_diff'] = df['ema_10'] - df['ema_50']
    return df

def _add_momentum_indicators(df):
    for n in [7, 14, 21]:
        delta = df['adj_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n, min_periods=1).mean()
        rs = gain.div(loss)
        df[f'rsi_{n}'] = 100 - (100 / (1 + rs))
    for n in [9, 14, 21]:
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        df[f'stoch_k_{n}'] = 100 * ((df['adj_close'] - low_n) / (high_n - low_n))
        df[f'stoch_d_{n}'] = df[f'stoch_k_{n}'].rolling(window=3).mean()
    ema_fast = df['adj_close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['adj_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    for n in [5, 10, 21, 63]:
        df[f'roc_{n}'] = df['adj_close'].pct_change(periods=n, fill_method=None) * 100
    return df

def _add_volatility_indicators(df):
    for n in [20, 50]:
        sma = df['adj_close'].rolling(window=n).mean()
        std = df['adj_close'].rolling(window=n).std()
        df[f'bb_upper_{n}'] = sma + (std * 2)
        df[f'bb_lower_{n}'] = sma - (std * 2)
        df[f'bb_width_{n}'] = ((df[f'bb_upper_{n}'] - df[f'bb_lower_{n}']) / sma) * 100
    for n in [10, 14, 20]:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['adj_close'].shift())
        low_close = np.abs(df['low'] - df['adj_close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df[f'atr_{n}'] = true_range.ewm(alpha=1/n, adjust=False).mean()
    for n in [10, 20, 50, 100]:
        df[f'volatility_{n}'] = df['adj_close'].pct_change(fill_method=None).rolling(n).std() * np.sqrt(252)
    return df

def _add_volume_indicators(df):
    df['obv'] = (np.sign(df['adj_close'].diff()) * df['volume']).fillna(0).cumsum()
    epsilon = 1e-10
    mfv = ((df['adj_close'] - df['low']) - (df['high'] - df['adj_close'])) / (df['high'] - df['low'] + epsilon)
    mfv = mfv.fillna(0) * df['volume']
    for n in [21, 50]:
        df[f'cmf_{n}'] = mfv.rolling(window=n).sum() / (df['volume'].rolling(window=n).sum() + epsilon)
    for n in [20, 50]:
        df[f'volume_sma_{n}'] = df['volume'].rolling(window=n).mean()
    return df

def _add_advanced_and_interaction_features(df):
    df['return_1d'] = df['adj_close'].pct_change(periods=1, fill_method=None)
    for feature in ['return_1d', 'rsi_14', 'volatility_20', 'macd']:
        if feature in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    for feature in ['rsi_14', 'stoch_k_14', 'macd']:
        if feature in df.columns:
            df[f'{feature}_rolling_mean_10'] = df[feature].rolling(10).mean()
            df[f'{feature}_rolling_std_10'] = df[feature].rolling(10).std()
    if 'bb_upper_20' in df.columns:
        df['close_vs_bb_upper'] = df['adj_close'] / df['bb_upper_20']
        df['close_vs_bb_lower'] = df['adj_close'] / df['bb_lower_20']
    if 'omx_close' in df.columns:
        df['omx_return_1d'] = df['omx_close'].pct_change(fill_method=None)
        df['relative_return_1d'] = df['return_1d'] - df['omx_return_1d']
    df['day_of_week'] = df.index.dayofweek
    df['month_of_year'] = df.index.month
    df['week_of_year'] = df.index.isocalendar().week.astype(float)
    return df

# =============================================================================
# HUVUDFUNKTIONER FÖR DATAPIPELINE
# =============================================================================

def _create_tables(conn):
    """Skapar databastabellen 'stocks_raw' om den inte finns."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks_raw (date TEXT, ticker TEXT, open REAL, 
        high REAL, low REAL, close REAL, adj_close REAL, volume INTEGER, 
        PRIMARY KEY (date, ticker))''')
    conn.commit()

def _fetch_raw_data(tickers, conn):
    """Hämtar rådata från yfinance."""
    print("--- Startar Steg 1: Hämtar rådata ---")
    
    symbols_to_fetch = tickers + ['^VIX', '^OMX']
    for symbol in symbols_to_fetch:
        db_ticker_name = 'OMXS30' if symbol == '^OMX' else ('VIX' if symbol == '^VIX' else symbol)
        print(f"Hämtar data för {db_ticker_name}...")
        
        start_date = DataConfig.START_DATE
        try:
            query = f"SELECT MAX(date) FROM stocks_raw WHERE ticker = '{db_ticker_name}'"
            max_date_db = pd.read_sql(query, conn).iloc[0, 0]
            if max_date_db:
                start_date_obj = datetime.strptime(max_date_db.split(' ')[0], '%Y-%m-%d')
                start_date = (start_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        except (pd.io.sql.DatabaseError, IndexError, TypeError): 
            pass  # Använd standardstartdatum

        if start_date >= DataConfig.END_DATE:
            print(f"Data för {db_ticker_name} är redan uppdaterad.")
            continue
            
        print(f"  Hämtar från {start_date} till {DataConfig.END_DATE}")
        data = yf.download(symbol, start=start_date, end=DataConfig.END_DATE, auto_adjust=False, progress=False)
        
        if data.empty:
            print(f"Ingen data hittades för {db_ticker_name}.")
            continue
            
        # FIX: Hantera multi-level kolumner från yfinance
        data.reset_index(inplace=True)
        
        # Debug: Visa kolumnnamn
        print(f"  Kolumner för {db_ticker_name}: {list(data.columns)}")
        
        # Hantera multi-level kolumner (tuples) från yfinance
        if isinstance(data.columns[0], tuple):
            # Platta ut multi-level kolumner - ta första delen av tupeln
            new_columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    # Ta första delen som inte är tom
                    new_name = col[0] if col[0] else col[1]
                    new_columns.append(new_name)
                else:
                    new_columns.append(col)
            data.columns = new_columns
        
        # Hitta datumkolumnen
        date_col = None
        for col in data.columns:
            col_lower = str(col).lower()
            if col_lower in ['date', 'index'] or 'date' in col_lower:
                if data[col].dtype.name.startswith('datetime') or col_lower == 'date':
                    date_col = col
                    break
        
        if date_col is None:
            print(f"Fel: Kunde inte hitta en datumkolumn för {db_ticker_name}. Tillgängliga kolumner: {list(data.columns)}")
            continue
        
        # Döp om datumkolumnen till 'date'
        data.rename(columns={date_col: 'date'}, inplace=True)
        
        # Standardisera alla kolumnnamn
        data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]
        
        # Formatera datum
        data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        data['ticker'] = db_ticker_name
        
        # Kontrollera att vi har de kolumner vi behöver
        required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        available_cols = ['date', 'ticker']
        
        for col in required_cols[1:]:  # Skippa 'date'
            if col in data.columns:
                available_cols.append(col)
        
        if len(available_cols) < 6:  # Minst date, ticker och 4 priskolumner
            print(f"Varning: {db_ticker_name} saknar nödvändiga kolumner. Har: {available_cols}")
            continue
        
        data_to_save = data[available_cols]
        data_to_save.to_sql('stocks_raw', conn, if_exists='append', index=False)
        print(f"  Sparade {len(data_to_save)} nya rader rådata för {db_ticker_name}.")

def _calculate_and_save_features(tickers, conn):
    """Kör hela feature-fabriken och sparar till databasen."""
    print("\n--- Startar Steg 2: Bygger Feature-databas ---")
    
    # Hämta VIX och OMX data och konvertera datum till string för merge
    vix_df = pd.read_sql("SELECT date, adj_close as vix_close FROM stocks_raw WHERE ticker='VIX' ORDER BY date", conn)
    omx_df = pd.read_sql("SELECT date, adj_close as omx_close FROM stocks_raw WHERE ticker='OMXS30' ORDER BY date", conn)
    
    # Konvertera datum till samma format som aktiedata kommer att ha
    if not vix_df.empty:
        vix_df['date'] = pd.to_datetime(vix_df['date']).dt.strftime('%Y-%m-%d')
    if not omx_df.empty:
        omx_df['date'] = pd.to_datetime(omx_df['date']).dt.strftime('%Y-%m-%d')

    all_prepared_dfs = []
    all_removed_tickers = []
    for ticker in tickers:
        print(f"Beräknar features för {ticker}...")
        df = pd.read_sql(f"SELECT * FROM stocks_raw WHERE ticker='{ticker}' ORDER BY date", conn)
        
        # Sanera rådata INNAN några beräkningar görs för att undvika TypeError.
        if df.empty:
            print(f"  - Ingen rådata för {ticker}. Skippar.")
            continue
        
        # Sanera och filtrera aktier som har för stor rörelse eller är penny stocks
        df, removed_tickers = clean_and_filter_data(df, price_col='adj_close')

        all_removed_tickers.extend(removed_tickers)

        if df.empty or len(df) < 200:
            print(f"  - För lite giltig data efter sanering för {ticker} ({len(df)} rader). Skippar.")
            continue

        # Konvertera alla kolumner som ska vara numeriska
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ta bort rader där kärndata saknas (t.ex. inget stängningspris)
        df.dropna(subset=['close', 'high', 'low'], inplace=True)
        
        if len(df) < 200:
            print(f"  - För lite giltig data efter sanering för {ticker} ({len(df)} rader). Skippar.")
            continue

        # FIX: Konvertera datum till konsistent format innan vi sätter index
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Kör igenom hela feature-fabriken
        df = _add_moving_averages(df)
        df = _add_momentum_indicators(df)
        df = _add_volatility_indicators(df)
        df = _add_volume_indicators(df)

        # Slå ihop med extern data - konvertera tillbaka till string för merge
        df.reset_index(inplace=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        if not vix_df.empty:
            df = pd.merge(df, vix_df, on='date', how='left', suffixes=('', '_vix'))
        if not omx_df.empty:
            df = pd.merge(df, omx_df, on='date', how='left', suffixes=('', '_omx'))
        
        # Fill missing external data
        if 'vix_close' in df.columns:
            df['vix_close'] = pd.to_numeric(df['vix_close'], errors='coerce').ffill().bfill()
        if 'omx_close' in df.columns:
            df['omx_close'] = pd.to_numeric(df['omx_close'], errors='coerce').ffill().bfill()
        
        # Konvertera tillbaka till datetime för advanced features
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = _add_advanced_and_interaction_features(df)
        
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        
        all_prepared_dfs.append(df)
        print(f"  - Klar! Skapade {len(df.columns)} features för {len(df)} rader.")

    if all_prepared_dfs:
        final_df = pd.concat(all_prepared_dfs, ignore_index=True)
        
        # FÖRBÄTTRAD RENSNING: Hantera NaN/inf mer intelligent
        print(f"\nAnalyserar datakvalitet...")
        initial_rows = len(final_df)
        
        # Ersätt inf värden med NaN
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Analysera vilka kolumner som har mest missing data
        missing_analysis = final_df.isnull().sum()
        problematic_features = missing_analysis[missing_analysis > len(final_df) * 0.5].index.tolist()
        
        if problematic_features:
            print(f"Tar bort {len(problematic_features)} features med >50% saknade värden: {problematic_features[:5]}{'...' if len(problematic_features) > 5 else ''}")
            final_df.drop(columns=problematic_features, inplace=True)
        
        # Identifiera kärnkolumner som MÅSTE finnas
        core_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        core_columns = [col for col in core_columns if col in final_df.columns]
        
        # Ta bara bort rader där kärndata saknas
        rows_before = len(final_df)
        final_df.dropna(subset=core_columns, inplace=True)
        rows_lost_core = rows_before - len(final_df)
        
        # För övriga features: fyll NaN med strategiska värden istället för att ta bort rader
        feature_columns = [col for col in final_df.columns if col not in ['date', 'ticker']]
        
        # Olika strategier för olika typer av features
        for col in feature_columns:
            if col in final_df.columns and final_df[col].isnull().any():
                if 'rsi' in col.lower():
                    # RSI: fyll med 50 (neutral)
                    final_df[col] = final_df[col].fillna(50)
                elif 'volume' in col.lower():
                    # Volym: fyll med median för den aktien
                    final_df[col] = final_df.groupby('ticker')[col].transform(lambda x: x.fillna(x.median()))
                elif any(indicator in col.lower() for indicator in ['sma', 'ema', 'bb']):
                    # Glidande medelvärden: forward fill sedan backward fill
                    final_df[col] = final_df.groupby('ticker')[col].transform(lambda x: x.ffill().bfill())
                else:
                    # Övriga: fyll med median
                    final_df[col] = final_df.groupby('ticker')[col].transform(lambda x: x.fillna(x.median()))
        
        # Final check: ta bort rader som fortfarande har NaN i kritiska kolumner
        final_rows_before = len(final_df)
        final_df.dropna(subset=['close', 'volume'], inplace=True)
        final_rows_lost = final_rows_before - len(final_df)
        
        print(f"Datarensning slutförd:")
        print(f"  - Startade med: {initial_rows:,} rader")
        print(f"  - Förlorade {rows_lost_core:,} rader pga saknad kärndata")
        print(f"  - Förlorade {final_rows_lost:,} rader i slutkontroll")
        print(f"  - Totalt kvar: {len(final_df):,} rader ({(len(final_df)/initial_rows)*100:.1f}%)")
        print(f"  - Totalt kvar: {len(final_df):,} rader ({(len(final_df)/initial_rows)*100:.1f}%)")
        print(f"  - Totalt borttagna tickers: {len(set(all_removed_tickers))}")


        final_df.to_sql('stocks_prepared', conn, if_exists='replace', index=False)
        print("\nDen nya 'stocks_prepared'-tabellen har sparats med alla nya features.")
    else:
        print("\nKunde inte skapa några features. Kontrollera rådatan.")

        
def run_data_pipeline(tickers):
    """Huvudfunktion som orkestrerar hela datainsamlings- och bearbetningsflödet."""
    db_name = DatabaseConfig.DB_NAME
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        _create_tables(conn)
        _fetch_raw_data(tickers, conn)
        _calculate_and_save_features(tickers, conn)
        print("\nDatapipelinen är färdig!")
    except Exception as e:
        print(f"Ett fel uppstod i datainsamlingen: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()