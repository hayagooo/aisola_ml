import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the TFLite model
tflite_model_path = "lstm_gru.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load original dataset for normalization
csv_file_path = 'data.csv'
data = pd.read_csv(csv_file_path)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / np.timedelta64(1, 's')
dataset = data[["waterlevel", "temperature", "vibration"]]

# Normalize the dataset
scaler = MinMaxScaler()
dataset_normalized = scaler.fit_transform(dataset)

# Prepare the last sequence for prediction
sequence_length = 20
last_sequence = dataset_normalized[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, dataset_normalized.shape[1])

# Predict for 1 week (7 days, 5-minute intervals)
one_week_intervals = int((7 * 24 * 60) / 5)
predictions = []
for _ in range(one_week_intervals):
    interpreter.set_tensor(input_details[0]['index'], last_sequence.astype(np.float32))
    interpreter.invoke()
    predicted = interpreter.get_tensor(output_details[0]['index'])
    scalar_prediction = predicted[0, -1, 0]  # Take the last timestep's prediction
    predictions.append(scalar_prediction)
    predicted_feature_vector = np.zeros((1, 1, last_sequence.shape[2]))
    predicted_feature_vector[0, 0, 0] = scalar_prediction
    new_input = np.append(last_sequence[:, 1:, :], predicted_feature_vector, axis=1)
    last_sequence = new_input

# Convert predictions to a NumPy array
predictions = np.array(predictions)

# Populate predicted_data with inverse-transformed predictions
predicted_data = np.zeros((len(predictions), dataset.shape[1]))
predicted_data[:, 0] = predictions
predicted_data = scaler.inverse_transform(predicted_data)

# Generate timestamps starting from tomorrow
start_date = pd.to_datetime(data['timestamp'].max()) + pd.Timedelta(days=1)
future_timestamps = pd.date_range(start=start_date, periods=one_week_intervals, freq='5T')

# Create a DataFrame for the predictions
predicted_df = pd.DataFrame(predicted_data, columns=["waterlevel", "temperature", "vibration"])
predicted_df['timestamp'] = future_timestamps

# Save the predictions to a CSV file
predicted_df.to_csv('predicted_1_week_tflite.csv', index=False)
print("1 Week predictions saved to 'predicted_1_week_tflite.csv'")

# Helper function to apply moving average smoothing
def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Plot Original Data (from CSV) and Predicted Data with Smoothing
plt.figure(figsize=(14, 7))

# Convert normalized timestamps in original data back to datetime
original_timestamps = pd.to_datetime(data['timestamp'] * np.timedelta64(1, 's') + pd.Timestamp(data['timestamp'].min()))

# Smooth original data
data['smoothed_waterlevel'] = smooth_data(data['waterlevel'], window_size=10)

# Smooth predicted data
predicted_df['smoothed_waterlevel'] = smooth_data(predicted_df['waterlevel'], window_size=10)

# Plot original water level data
plt.plot(original_timestamps, data['smoothed_waterlevel'], label="Original Data (Smoothed)", color="blue", alpha=0.7)

# Plot predicted water level data
plt.plot(predicted_df['timestamp'], predicted_df['smoothed_waterlevel'], label="Predicted Data (Smoothed)", color="red", alpha=0.7)

# Add plot details
plt.title("Water Level Prediction vs Original Data (Smoothed)")
plt.xlabel("Time")
plt.ylabel("Water Level (cm)")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
