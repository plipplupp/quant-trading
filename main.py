from config import DataConfig
from data_pipeline import run_data_pipeline
from generate_targets import generate_targets
from train_models import train_models
from backtest_regression import backtest_regression
from backtest_binary import backtest_binary
from backtest_ranking import backtest_ranking
import os
import matplotlib.pyplot as plt
import pandas as pd
from config import PathsConfig
from load_models import load_models

# Ladda alla modeller
models = load_models()

def show_results():
    print("\n--- Sammanfattande resultat ---")

    files = {
        "Regression": "regression_results.csv",
        "Binary": "binary_results.csv",
        "Ranking": "ranking_results.csv"
    }

    portfolios = {}
    for name, file in files.items():
        path = os.path.join(PathsConfig.RESULTS_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "portfolio" in df.columns:
                portfolios[name] = df
            else:
                portfolios[name] = pd.DataFrame({"portfolio": df.iloc[:,1].values})
            print(f"{name}: slutkapital {portfolios[name]['portfolio'].iloc[-1]:.2f} kr")
        else:
            print(f"{name}: inget resultat hittat.")

    if portfolios:
        plt.figure(figsize=(10,6))
        for name, df in portfolios.items():
            plt.plot(df['portfolio'].values, label=name)
        plt.title("Portföljutveckling")
        plt.legend()
        plt.show()

def run_all_backtests():
    backtest_regression()
    backtest_binary()
    backtest_ranking()
    show_results()

def main():
    while True:
        print("\n--- MENY ---")
        print("1. Uppdatera data + features")
        print("2. Generera målvariabler")
        print("3. Träna modeller")
        print("4. Backtest Regression")
        print("5. Backtest Binary vs Index")
        print("6. Backtest Ranking")
        print("7. Backtest Alla Strategier")
        print("8. Visa resultat")
        print("0. Avsluta")

        choice = input("Välj ett alternativ: ")

        if choice == "1":
            run_data_pipeline(DataConfig.TICKERS)
        elif choice == "2":
            generate_targets()
        elif choice == "3":
            train_models()
        elif choice == "4":
            backtest_regression()
        elif choice == "5":
            backtest_binary()
        elif choice == "6":
            backtest_ranking()
        elif choice == "7":
            run_all_backtests()
        elif choice == "8":
            show_results()
        elif choice == "0":
            break
        else:
            print("Fel val. Försök igen.")

if __name__ == "__main__":
    main()
