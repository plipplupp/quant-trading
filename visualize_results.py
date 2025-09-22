import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import PathsConfig, DataConfig, BacktestConfig
from data_pipeline import run_data_pipeline
from generate_targets import generate_targets
from train_models import train_models
from backtest_regression import backtest_regression
from backtest_binary import backtest_binary
from backtest_ranking import backtest_ranking
from benchmark_analysis import run_omx_benchmark, run_equal_weight_benchmark
from load_models import load_models


def show_results():
    """
    Läser in alla dagliga resultat-filer (strategier och benchmarks),
    beräknar procentuell avkastning och plottar en jämförande graf.
    """
    print("\n--- Sammanställning och visualisering av resultat ---")

    # Definiera alla strategier och deras respektive resultatfiler
    strategies = {
        "OMXS30 Benchmark": "benchmark_omx_daily.csv",
        "Lika Viktad Portfölj": "benchmark_equal_weight_daily.csv",
        "Regression-Strategi": "regression_daily.csv",
        "Binary-Strategi": "binary_daily.csv",
        #"Ranking-Strategi": "ranking_daily.csv"
    }

    all_portfolios_df = pd.DataFrame()

    # Läs in och slå samman alla resultatfiler som finns
    for name, filename in strategies.items():
        path = os.path.join(PathsConfig.RESULTS_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Säkerställ att kolumnerna har rätt namn och format
                df.rename(columns={'portfolio': 'portfolio_value'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Konvertera till numeriskt värde om det är strängar.
                # Denna rad kan du ta bort helt om du är säker på att alla filer sparas med floats.
                df['portfolio_value'] = pd.to_numeric(df['portfolio_value'].astype(str).str.replace(' ', '').str.replace(',', ''), errors='coerce')

                # Lägg till portföljens utveckling som en ny kolumn
                all_portfolios_df[name] = df['portfolio_value']
                print(f"✅ Laddade resultat för: {name}")
            except Exception as e:
                print(f"⚠️ Kunde inte ladda filen för {name}: {e}")
        else:
            print(f"ℹ️ Fil för '{name}' hittades inte, hoppar över.")

    if all_portfolios_df.empty:
        print("\nInga resultatfiler hittades att visa. Kör backtester och benchmarks först.")
        return

    # Fyll i saknade värden (helgdagar etc.) genom att ta föregående dags värde
    all_portfolios_df.ffill(inplace=True)

    # Beräkna och skriv ut procentuell utveckling
    print("\n--- Procentuell utveckling ---")
    initial_capital = float(BacktestConfig.INITIAL_CAPITAL)
    
    # Sortera för snyggare utskrift
    final_returns = {}
    for name in all_portfolios_df.columns:
        start_value = all_portfolios_df[name].dropna().iloc[0]
        end_value = all_portfolios_df[name].dropna().iloc[-1]
        percent_return = ((end_value / start_value) - 1) * 100
        final_returns[name] = percent_return

    sorted_returns = sorted(final_returns.items(), key=lambda item: item[1], reverse=True)

    for name, percent_return in sorted_returns:
        print(f"{name+':':<25} {percent_return:+.2f}%")

    # Skapa grafen
    plt.style.use('seaborn-v0_8-whitegrid') # Uppdaterad stil
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_portfolios_df.plot(ax=ax, linewidth=2)
    
    # Formatera grafen för att göra den snygg och lättläst
    ax.set_title("Jämförelse av portföljutveckling", fontsize=16)
    ax.set_ylabel("Portföljvärde (kr)", fontsize=12)
    ax.set_xlabel("Datum", fontsize=12)
    
    # Formatera y-axeln för att visa tusentalsavgränsare
    ax.get_yaxis().set_major_formatter(
        mticker.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    plt.legend(title="Strategier", fontsize=10)
    plt.tight_layout()
    plt.show()