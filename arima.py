import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import pmdarima as pm

df = pd.read_csv('data.csv')

result = adfuller(df.kwh.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.kwh); axes[0, 0].set_title('Original Series')
plot_acf(df.kwh, ax=axes[0, 1])

axes[1, 0].plot(df.kwh.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.kwh.diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(df.kwh.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.kwh.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.kwh.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.kwh.diff().dropna(), ax=axes[1])

plt.show()

model_112 = ARIMA(df.kwh, order=(1, 1, 2))
model_fit_112 = model_112.fit()
print("ARIMA(1,1,2) Summary")
print(model_fit_112.summary())

model_111 = ARIMA(df.kwh, order=(1, 1, 1))
model_fit_111 = model_111.fit()
print("ARIMA(1,1,1) Summary")
print(model_fit_111.summary())

residuals = pd.DataFrame(model_fit_111.resid, columns=['Residuals'])
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

pred = model_fit_111.get_prediction(start=0, end=len(df.kwh)-1, dynamic=False)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(df.kwh, label="Actual")
plt.plot(pred_mean, label="Predicted", color='orange')
plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title("Actual vs Predicted")
plt.show()

train = df.kwh[:3000]
test = df.kwh[3000:]

model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()

forecast = fitted.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05) 

plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Data')
plt.plot(forecast_mean, label='Forecast', color='orange')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title("Forecast vs Actuals")
plt.legend(loc='upper left', fontsize=10)
plt.show()


# model = ARIMA(train, order=(3, 2, 1))
# fitted = model.fit()

# forecast = fitted.get_forecast(steps=len(test))
# forecast_mean = forecast.predicted_mean
# forecast_ci = forecast.conf_int(alpha=0.05) 

# plt.figure(figsize=(12, 6))
# plt.plot(train, label='Training Data')
# plt.plot(test, label='Actual Data')
# plt.plot(forecast_mean, label='Forecast', color='orange')
# plt.fill_between(forecast_ci.index, 
#                  forecast_ci.iloc[:, 0], 
#                  forecast_ci.iloc[:, 1], 
#                  color='pink', alpha=0.3, label='Confidence Interval')
# plt.title("Forecast vs Actuals")
# plt.legend(loc='upper left', fontsize=10)
# plt.show()

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100  # Mean Absolute Percentage Error
    me = np.mean(forecast - actual)                                  # Mean Error
    mae = np.mean(np.abs(forecast - actual))                         # Mean Absolute Error
    mpe = np.mean((forecast - actual) / actual) * 100                # Mean Percentage Error
    rmse = np.sqrt(np.mean((forecast - actual)**2))                  # Root Mean Squared Error
    corr = np.corrcoef(forecast, actual)[0, 1]                       # Correlation coefficient
    mins = np.amin(np.vstack([forecast, actual]), axis=0)
    maxs = np.amax(np.vstack([forecast, actual]), axis=0)
    minmax = 1 - np.mean(mins / maxs)                                # Min-Max Error

    residuals = actual - forecast
    acf1 = acf(residuals, nlags=1)[1]                                # Autocorrelation of residuals at lag 1

    return {
        'mape': mape,
        'me': me,
        'mae': mae,
        'mpe': mpe,
        'rmse': rmse,
        'acf1': acf1,
        'corr': corr,
        'minmax': minmax
    }
    
forecast = model_fit_111.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean

accuracy_metrics = forecast_accuracy(forecast_mean.values, test.values)
print("Forecast Accuracy Metrics:")
for metric, value in accuracy_metrics.items():
    print(f"{metric}: {value}")
    
model = pm.auto_arima(df.kwh, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(10,8))
plt.show()

# n_periods = 2000
# fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
# index_of_fc = np.arange(len(df.kwh), len(df.kwh)+n_periods)

# fc_series = pd.Series(fc, index=index_of_fc)
# lower_series = pd.Series(confint[:, 0], index=index_of_fc)
# upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# plt.plot(df.kwh)
# plt.plot(fc_series, color='darkgreen')
# plt.fill_between(lower_series.index,
#                  lower_series,
#                  upper_series,
#                  color='k', alpha=.15)

# plt.title("Final Forecast of Usage")
# plt.show()