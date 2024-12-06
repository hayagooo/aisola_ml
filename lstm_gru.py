import tensorflow as tf
from tensorflow.keras.layers import Attention, Input, Dropout, Bidirectional, Dense, LSTM, GRU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'data.csv'
data = pd.read_csv(csv_file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = (data['timestamp'] - data['timestamp'].min()) / np.timedelta64(1, 's')
dataset = data[["waterlevel", "temperature", "vibration"]]
scaler = MinMaxScaler()
dataset_normalized = scaler.fit_transform(dataset)

def create_sequences(data, sequence_length=20):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])
    return np.array(X), np.array(y)

sequence_length = 20
X, y = create_sequences(dataset_normalized, sequence_length)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

class AttentionLayer(tf.keras.layers.Layer):
    def call(self, query, value):
        scores = tf.matmul(query, value, transpose_b=True)
        distribution = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(distribution, value)
        return context

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    gru_out = Bidirectional(GRU(128, activation='relu', return_sequences=False))(lstm_out) 
    dense_out = Dense(64, activation='relu')(gru_out)
    dense_out = Dropout(0.2)(dense_out)
    output = Dense(1)(dense_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

hybrid_model = build_hybrid_model((sequence_length, X.shape[2]))
history = hybrid_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

test_loss, test_mae = hybrid_model.evaluate(X_test, y_test, verbose=0)
converter = tf.lite.TFLiteConverter.from_keras_model(hybrid_model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

tflite_model_path = "lstm_gru.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Hybrid model saved as {tflite_model_path}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()

hybrid_model.summary()