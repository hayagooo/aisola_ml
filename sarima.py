import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

data = pd.read_csv('data.csv', parse_dates=['timestamp'], index_col='timestamp')
data = data.resample('1h').mean()
print(data.head())
target_column = 'kwh'
data = data[target_column]
data = data.fillna(data.median())

fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Aitoma Sensors - Time Series Dataset', fontsize=16)
plt.show()

smodel = pm.auto_arima(data, 
    start_p=1, 
    start_q=1,
    test='adf',
    max_p=2,
    max_q=2, 
    m=24,
    seasonal=True,
    d=None, D=1, trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

smodel.summary()

n_periods = 7 * 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(data.index[-1], periods = n_periods, freq='H')

fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.plot(data)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("SARIMA - Aitoma Sensors - Time Series Dataset")
plt.show()