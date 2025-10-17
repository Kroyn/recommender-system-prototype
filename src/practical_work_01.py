import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp
import numpy as np

data = sm.datasets.co2.load_pandas().data

df = data['co2']['1980':'1990'].resample('M').mean().to_frame()

decomposition = seasonal_decompose(df['co2'], model='additive')

def func1():
    plt.figure(figsize=(12, 6))
    plt.plot(df['co2'])
    plt.title('Time series graph (CO2 concentration)')
    plt.xlabel('Date')
    plt.ylabel('CO2 concentration')
    plt.grid(True)
    plt.show()

def func2():
    print("Visualization of time series components:")
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.show()

def func3():
    residual = decomposition.resid.dropna()

    print("\nWe construct a graph of the autocorrelation function for the residuals:")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_acf(residual, ax=ax, lags=20)
    plt.title('Autocorrelation function of the random component (residuals)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()

def func4():
    print("\n--- Trend test (series criterion) ---")

    diffs = df['co2'].diff().dropna()
    signs = np.sign(diffs)
    signs = signs[signs != 0]

    stat, p_value = runstest_1samp(signs, correction=False)

    print(f"P-value of the test: {p_value:.5f}")

    if p_value < 0.05:
        print("Result: The hypothesis of randomness is rejected. There is a trend in the series.")
    else:
        print("Result: The hypothesis of randomness is not rejected. The trend is not statistically confirmed.")

if __name__ == "__main__":
    if df is not None:
        while True:
            print("\n=======================================================")
            print("Select the functions:")
            print("(1) Visualization of the time series")
            print("(2) Decomposition of the series into components")
            print("(3) ACF of residuals")
            print("(4) Trend test (runs test)")
            print("(0) Exit")
            print("\n=======================================================")

            select = input("Enter number: ")

            match select:
                case '1':
                    func1()
                case '2':
                    func2()
                case '3':
                    func3()
                case '4':
                    func4()
                case '0':
                    break
                case _:
                    print("Incorrect selection. Please try again.")

