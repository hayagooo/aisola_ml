import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.utils import plot_model

# Membaca dataset dan preprocessing
data = pd.read_csv("data.csv", na_values=['nan', '?'])
data['datetime'] = pd.to_datetime(data['timestamp'])
data.set_index('datetime', inplace=True)
data.drop('timestamp', axis=1, inplace=True)
data.fillna(0, inplace=True)
data_values = data.values.astype('float32')
print(data.head(4))

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data_values)
normalized_df = pd.DataFrame(normalized_data)
print(normalized_df.head(4))

# Membuat dataset untuk time series
def prepare_timeseries(dataset, look_back=100, target_col=-1):
    temp_data = dataset.copy()
    temp_data["row_id"] = range(1, len(temp_data) + 1)
    temp_data = temp_data.iloc[:-look_back, :]
    temp_data.set_index('row_id', inplace=True)

    target_data = dataset.copy().iloc[look_back:, target_col]
    target_data = pd.DataFrame(target_data, columns=["target"])
    target_data["row_id"] = range(1, len(target_data) + 1)
    target_data.set_index('row_id', inplace=True)

    combined_df = pd.concat([temp_data, target_data], axis=1)
    return combined_df

processed_data = prepare_timeseries(normalized_df, 1, 3)
processed_data.fillna(0, inplace=True)
processed_data.columns = ['feature1(t-1)', 'feature2(t-1)', 'feature3(t-1)', 'feature4(t-1)', 'target(t)']
print(processed_data.head())

# Statistik data
column_means = processed_data.mean()
print(column_means)

# Membagi dataset menjadi data latih dan uji
values = processed_data.values
train_size = int(len(data) * 0.7)
train_data, test_data = values[:train_size, :], values[train_size:, :]
train_X, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

# Reshape input untuk LSTM
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Model LSTM
model = Sequential([
    LSTM(75, return_sequences=True, input_shape=(1, train_X.shape[2])),
    LSTM(30, return_sequences=True),
    LSTM(30),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae')
history = model.fit(train_X, train_y, epochs=10, validation_data=(test_X, test_y), batch_size=64, shuffle=False)
predictions = model.predict(test_X)

# Visualisasi hasil pelatihan
plt.plot(history.history['loss'], label='Training Loss', color='brown')
plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
plt.legend()
plt.title('Loss Curve')
plt.show()

# Evaluasi model
mse = mean_squared_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test_y, predictions)
print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

plt.plot(test_y, label='Actual Values', color='blue')
plt.plot(predictions, label='Predicted Values', color='orange')
plt.legend()
plt.title('Actual vs Predicted')
plt.show()

# Simpan model ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("forecast_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil disimpan dalam format TFLite!")

# Visualisasi tambahan
plot_model(model, to_file='model_architecture.png', show_shapes=True, expand_nested=True)
print("Arsitektur model disimpan sebagai 'model_architecture.png'.")