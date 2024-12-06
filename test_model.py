import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = pd.read_csv("data.csv", na_values=['nan', '?'])
dataset['dt'] = pd.to_datetime(dataset['timestamp'])
dataset.set_index('dt', inplace=True)
dataset.drop('timestamp', axis=1, inplace=True)
dataset.fillna(0, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.values)
scaled = pd.DataFrame(scaled, index=dataset.index, columns=dataset.columns)

interpreter = tf.lite.Interpreter(model_path="aisola_lstm.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

lookback = 1
last_data = scaled.values[-lookback:, :].reshape((1, lookback, scaled.shape[1]))

future_predictions = []
for _ in range(2016):
    interpreter.set_tensor(input_details[0]['index'], last_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    future_predictions.append(output_data[0][0])

    predicted_row = np.zeros((1, scaled.shape[1]))
    predicted_row[0, -1] = output_data[0][0] 
    
    next_input = np.vstack([last_data[0, 1:, :], predicted_row])
    last_data = next_input.reshape((1, lookback, scaled.shape[1]))

future_predictions = np.array(future_predictions).reshape(-1, 1)
denormalized_predictions = scaler.inverse_transform(
    np.hstack([np.zeros((future_predictions.shape[0], scaled.shape[1] - 1)), future_predictions])
)[:, -1]

last_timestamp = dataset.index[-1]
future_timestamps = pd.date_range(last_timestamp, periods=2016, freq="5min")

future_df = pd.DataFrame({
    "Predicted_kwh": denormalized_predictions
}, index=future_timestamps)

future_df.to_csv("future_predictions_lstm.csv")

plt.figure(figsize=(15, 6))
plt.plot(dataset['kwh'], label='Actual kwh')
plt.plot(future_df['Predicted_kwh'], label='Predicted kwh (7 days)', color='orange')
plt.legend()
plt.title('Actual vs Predicted kwh (7 days ahead)')
plt.show()
