# Aisola Model Prediction
## Time Series Analysis and Forecasting
This project demonstrates the application of LSTM, GRU, ARIMA, and SARIMA models for time series forecasting. The dataset used represents energy consumption (kWh) over time, and predictions are made for the next 7 days with a 5-minute interval.

### 1. Long Short-Term Memory (LSTM)
LSTM is a type of recurrent neural network (RNN) well-suited for time series data due to its ability to retain long-term dependencies.
#### Key Result
- Actual Data & Predicted Data
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/lstm/result4.png?raw=true)
- Residual Distribution
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/lstm/residual.png?raw=true)
#### Evaluation
- Training and Validation Loss
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/lstm/loss.png?raw=true)
- Actual vs Predicted Scatter Plot
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/lstm/predicted_scatter.png?raw=true)
- Bar Chart of Loss
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/lstm/loss_bar_chart.png?raw=true)

### 2. Gated Recurrent Unit (GRU)
GRU is a simpler alternative to LSTM, reducing computational complexity while maintaining performance.
#### Key Result
- Actual Data & Predicted Data
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/gru/result4.png?raw=true)
- Residual Distribution
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/gru/residual.png?raw=true)
#### Evaluation
- Training and Validation Loss
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/gru/loss.png?raw=true)
- Actual vs Predicted Scatter Plot
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/gru/loss_scatter.png?raw=true)
- Bar Chart of Loss
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/gru/loss_bar_chart.png?raw=true)

### 3. AutoRegressive Integrated Moving Average (ARIMA)
ARIMA is a classic statistical model for time series data that handles non-stationary data with differencing.
#### Key Result
- Original data vs Differencing ACF
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/arima/Original_vs_Differencing_ACF.png?raw=true)
- Autocorrelation Plot
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/arima/residual_density.png?raw=true)
- Residual Density
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/arima/autocorellation.png?raw=true)
- Actual vs Predicted
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/arima/actual_vs_predicted.png?raw=true)
- Histogram estimated density, Correlogram, Norm Q-Q
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/arima/spesification.png?raw=true)

### 4. Seasonal AutoRegressive Integrated Moving Average (SARIMA)
SARIMA extends ARIMA by including seasonal differencing and seasonal lags, making it suitable for periodic time series.
#### Key Result
- Data
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/sarima/data.png?raw=true)
- Seasonal Result
  
  ![Image](https://github.com/hayagooo/aisola_ml/blob/main/results/sarima/result.png?raw=true)
