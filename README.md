### Developed by : M.Pranathi
### Register no : 212222240064
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm

# Assuming the dataset is stored in a CSV file
# Load the dataset
data = pd.read_csv('supermarketsales.csv')

# Extract the 'Total' column to use as a time series
time_series = data['Total']

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(time_series, label='Total Sales')
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Total')
plt.legend()
plt.show()

# Simulate ARMA(1,1) process
# ARMA(1,1) process parameters: AR = [1, -0.5], MA = [1, 0.5]
ar1 = np.array([1, -0.5])  # AR coefficients
ma1 = np.array([1, 0.5])   # MA coefficients
arma11_process = ArmaProcess(ar1, ma1)
simulated_arma11 = arma11_process.generate_sample(nsample=len(time_series))

# Plot the simulated ARMA(1,1) process
plt.figure(figsize=(10, 4))
plt.plot(simulated_arma11, label='Simulated ARMA(1,1)', color='orange')
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Simulate ARMA(2,2) process
# ARMA(2,2) process parameters: AR = [1, -0.75, 0.25], MA = [1, 0.65, 0.35]
ar2 = np.array([1, -0.75, 0.25])  # AR coefficients
ma2 = np.array([1, 0.65, 0.35])   # MA coefficients
arma22_process = ArmaProcess(ar2, ma2)
simulated_arma22 = arma22_process.generate_sample(nsample=len(time_series))

# Plot the simulated ARMA(2,2) process
plt.figure(figsize=(10, 4))
plt.plot(simulated_arma22, label='Simulated ARMA(2,2)', color='green')
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Fitting ARMA(1,1) on the original time series using ARIMA with d=0
model_arma11 = sm.tsa.ARIMA(time_series, order=(1, 0, 1))
results_arma11 = model_arma11.fit()
print('ARMA(1,1) Model Summary:')
print(results_arma11.summary())

# Fitting ARMA(2,2) on the original time series using ARIMA with d=0
model_arma22 = sm.tsa.ARIMA(time_series, order=(2, 0, 2))
results_arma22 = model_arma22.fit()
print('ARMA(2,2) Model Summary:')
print(results_arma22.summary())

# Plot Autocorrelation Function (ACF) for Simulated ARMA(1,1)
plt.figure(figsize=(10, 4))
plot_acf(simulated_arma11, lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF) for Simulated ARMA(1,1)')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

# Plot Partial Autocorrelation Function (PACF) for Simulated ARMA(1,1)
plt.figure(figsize=(10, 4))
plot_pacf(simulated_arma11, lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) for Simulated ARMA(1,1)')
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.show()

# Plot Autocorrelation Function (ACF) for Simulated ARMA(2,2)
plt.figure(figsize=(10, 4))
plot_acf(simulated_arma22, lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF) for Simulated ARMA(2,2)')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

# Plot Partial Autocorrelation Function (PACF) for Simulated ARMA(2,2)
plt.figure(figsize=(10, 4))
plot_pacf(simulated_arma22, lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) for Simulated ARMA(2,2)')
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.show()

```

### OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/user-attachments/assets/193209ea-6b73-49a1-a83a-8288e719d6f5)

Partial Autocorrelation :

![image](https://github.com/user-attachments/assets/b46931cb-a288-4b79-afe1-997c1d17fb20)

Autocorrelation :

![image](https://github.com/user-attachments/assets/84c0be38-4b36-4163-9956-e20a2f1b9b56)


SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/user-attachments/assets/ad73e574-fac6-4f5b-8d27-d8bce571bb29)

Partial Autocorrelation :

![image](https://github.com/user-attachments/assets/fa18ee9d-92c1-453d-82ba-2f981d3075aa)


Autocorrelation:

![image](https://github.com/user-attachments/assets/4c14287d-1c9c-4c0a-8021-7450bb51dc57)


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
