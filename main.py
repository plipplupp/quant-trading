from config import DataConfig
from data_pipeline import run_data_pipeline
from generate_targets import generate_targets
import visualize_results
from train_models import train_models
from backtest_regression import backtest_regression
from backtest_binary import backtest_binary
from backtest_ranking import backtest_ranking
from benchmark_analysis import run_omx_benchmark, run_equal_weight_benchmark
from load_models import load_models

# Ladda alla modeller vid start
models = load_models()

def run_all_backtests():
    backtest_regression()
    backtest_binary()
    backtest_ranking()
    visualize_results.show_results()


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
        print("8. Kör Benchmark-analyser")
        print("9. Visa resultat")
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
            run_omx_benchmark()
            run_equal_weight_benchmark()
        elif choice == "9":
            visualize_results.show_results()
        elif choice == "0":
            break
        else:
            print("Fel val. Försök igen.")

if __name__ == "__main__":
    main()