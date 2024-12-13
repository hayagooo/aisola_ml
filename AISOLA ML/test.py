import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv("data.csv", na_values=['nan', '?'])
data['datetime'] = pd.to_datetime(data['timestamp'])
data.set_index('datetime', inplace=True)
data.drop(columns=['timestamp'], inplace=True)
data.fillna(0, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)
scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="aisola_lstm.tflite")
interpreter.allocate_tensors()

# Retrieve input and output tensor details
input_tensor = interpreter.get_input_details()
output_tensor = interpreter.get_output_details()

# Initialize variables for prediction
time_steps = 1
input_sequence = scaled_df.values[-time_steps:, :].reshape((1, time_steps, scaled_df.shape[1]))

predicted_values = []
num_predictions = 2016  # Number of future time steps

for _ in range(num_predictions):
    interpreter.set_tensor(input_tensor[0]['index'], input_sequence.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_tensor[0]['index'])
    predicted_values.append(prediction[0, 0])

    next_step = np.zeros((1, scaled_df.shape[1]))
    next_step[0, -1] = prediction[0, 0]
    input_sequence = np.vstack([input_sequence[0, 1:, :], next_step]).reshape((1, time_steps, scaled_df.shape[1]))

# Post-process predictions
predicted_array = np.array(predicted_values).reshape(-1, 1)
descaled_predictions = scaler.inverse_transform(
    np.hstack([np.zeros((predicted_array.shape[0], scaled_df.shape[1] - 1)), predicted_array])
)[:, -1]

# Create a DataFrame for future predictions
last_index = data.index[-1]
future_time_index = pd.date_range(start=last_index, periods=num_predictions + 1, freq="5min")[1:]

forecast_df = pd.DataFrame({
    "Forecast_kwh": descaled_predictions
}, index=future_time_index)

# Save predictions to a CSV file
forecast_df.to_csv("forecast_results_lstm.csv")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(data['kwh'], label='Observed kWh', color='blue')
plt.plot(forecast_df['Forecast_kwh'], label='Forecasted kWh (7 Days Ahead)', color='red')
plt.title('Energy Consumption Forecasting')
plt.xlabel('Timestamp')
plt.ylabel('kWh')
plt.legend()
plt.grid()
plt.show()
