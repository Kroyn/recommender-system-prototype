import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    data = sm.datasets.co2.load_pandas().data
    df = data['co2']['1980':'1990'].resample('M').mean().to_frame()
    df = df.fillna(df.bfill()) # Filling in the blanks
    df_log = np.log(df['co2'])
    df_seasonal_diff = df_log.diff().diff(12).dropna()
    model_results = None # Тут буде зберігатись навчена модель
    print("Data has been successfully loaded and prepared.")
except Exception as e:
    print(f"Data loading error: {e}")
    df = None # If the data is not loaded, the program will not be able to work.

def step1_check_stationarity():
    if df is None:
        print("Error: Data not loaded.")
        return

    def _adfuller_test(timeseries, series_name):
        print(f"\n--- Dickey-Fuller test results for '{series_name}' ---")
        result = adfuller(timeseries)
        p_value = result[1]
        print(f'P-value: {p_value:.6f}')
        if p_value <= 0.05:
            print("Conclusion: The series is stationary.")
        else:
            print("Conclusion: The series is not stationary.")

    _adfuller_test(df['co2'], "Starting row")

    plt.figure(figsize=(12, 6))
    plt.plot(df_seasonal_diff)
    plt.title('Stationary time series after transformations')
    plt.grid(True)
    plt.show()

    _adfuller_test(df_seasonal_diff, "Transformed row")

def step2_identify_model():
    if df is None:
        print("Error: Data not loaded.")
        return

    print("\n--- We build ACF and PACF to identify model parameters ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df_seasonal_diff, ax=ax1, lags=30)
    ax1.set_title('Autocorrelation function (ACF)')
    plot_pacf(df_seasonal_diff, ax=ax2, lags=30)
    ax2.set_title('Partial autocorrelation function (PACF)')
    plt.tight_layout()
    plt.show()

def step3_fit_and_diagnose():
    global model_results
    if df is None:
        print("Error: data not loaded.")
        return

    print("\n--- Training the model SARIMAX(1,1,1)(1,1,1,12) ---")
    model = SARIMAX(df_log, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_results = model.fit(disp=False)

    print("\n--- Step 4: Model parameters ---")
    print(model_results.summary())

    print("\n--- Step 5: Diagnostic graphs to check adequacy ---")
    model_results.plot_diagnostics(figsize=(15, 12))
    plt.tight_layout()
    plt.show()
    print("The model has been successfully trained and tested.")

def step4_forecast():
    if model_results is None:
        print("\nERROR: Model not yet trained. Please complete step first (3).")
        return

    print("\n--- Step 6: Build a forecast for 10 periods ahead ---")
    forecast_object = model_results.get_forecast(steps=10)
    forecast = forecast_object.predicted_mean
    confidence_intervals = forecast_object.conf_int()

    forecast_exp = np.exp(forecast)
    confidence_intervals_exp = np.exp(confidence_intervals)

    plt.figure(figsize=(14, 7))
    plt.plot(np.exp(df_log['1988':]), label='Observation')
    plt.plot(forecast_exp, label='Forecast', color='red', linestyle='--')
    plt.fill_between(confidence_intervals_exp.index,
                     confidence_intervals_exp.iloc[:, 0],
                     confidence_intervals_exp.iloc[:, 1],
                     color='pink', alpha=0.5, label='95% Confidence interval')
    plt.title('CO2 concentration forecast for 10 periods')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Головний цикл програми ---
if __name__ == "__main__":
    if df is not None:
        while True:
            print("\n=======================================================")
            print("Select the functions:")
            print("(1) Step 1-2: Checking the stationarity of the series")
            print("(2) Step 3: Model Identification (ACF/PACF)")
            print("(3) Step 4-5: Training and diagnosing the SARIMA model")
            print("(4) Step 6: Building a forecast for 10 periods")
            print("(0) Exit")
            print("=======================================================")

            select = input("Enter number: ")

            match select:
                case '1':
                    step1_check_stationarity()
                case '2':
                    step2_identify_model()
                case '3':
                    step3_fit_and_diagnose()
                case '4':
                    step4_forecast()
                case '0':
                    break
                case _:
                    print("Incorrect selection. Please try again.")
