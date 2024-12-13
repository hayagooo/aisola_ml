import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, GRU
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.utils import plot_model

# Load dataset
data = read_csv("data.csv", na_values=['nan', '?'])
data['datetime'] = pd.to_datetime(data['timestamp'])
data.set_index('datetime', inplace=True)
data.drop(columns=['timestamp'], inplace=True)
data.fillna(0, inplace=True)
data_values = data.values.astype('float32')
print(data.head(4))

# Normalize data
normalizer = MinMaxScaler(feature_range=(0, 1))
normalized_data = normalizer.fit_transform(data_values)
normalized_data = pd.DataFrame(normalized_data)
print(normalized_data.head(4))

# Function to prepare time series data
def generate_time_series(dataframe, lookback=2016, target_col=4):
    base_df = dataframe.copy()
    base_df['index'] = range(len(base_df))
    base_df = base_df.iloc[:-lookback, :]
    base_df.set_index('index', inplace=True)

    target_values = dataframe.iloc[lookback:, target_col]
    target_df = pd.DataFrame(target_values)
    target_df.columns = ['Target']
    target_df['index'] = range(len(target_df))
    target_df.set_index('index', inplace=True)

    result = pd.concat([base_df, target_df], axis=1)
    return result

reframed = generate_time_series(normalized_data, 1, 3)
reframed.fillna(0, inplace=True)
reframed.columns = ['Feature1(t-1)', 'Feature2(t-1)', 'Feature3(t-1)', 'Feature4(t-1)', 'Target(t)']
print(reframed.head())

# Data splitting and reshaping
mean_values = reframed.mean()
print(mean_values)

full_values = reframed.values
split_point = int(len(data) * 0.7)
train_data, test_data = full_values[:split_point, :], full_values[split_point:, :]

train_X, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Build and compile GRU model
gru_model = Sequential([
    GRU(75, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])),
    GRU(30, return_sequences=True),
    GRU(30),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mae')

# Train the model
history = gru_model.fit(
    train_X, train_y, 
    epochs=10, 
    validation_data=(test_X, test_y), 
    batch_size=64, 
    shuffle=False
)

# Make predictions
predictions = gru_model.predict(test_X)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss', color='orange')
plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
plt.legend()
plt.title('Model Loss')
plt.show()

# Evaluate the model
mse = mean_squared_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test_y, predictions)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Plot actual vs predicted
plt.figure(figsize=(15, 5))
plt.plot(test_y, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.legend()
plt.title('Actual vs Predicted')
plt.show()

# Save model as TFLite
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(gru_model)
tflite_converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = tflite_converter.convert()
with open("energy_gru_model.tflite", "wb") as tflite_file:
    tflite_file.write(tflite_model)
print("Model has been successfully converted to TFLite format.")

# Additional plots
residuals = test_y - predictions.flatten()
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residual Distribution')
plt.show()

plt.scatter(test_y, predictions, alpha=0.5, edgecolor='k')
plt.title('Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()