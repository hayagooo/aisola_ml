import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.optimizers import Adam
import tensorflow as tf
import optuna

# Load dataset
data = pd.read_csv("data.csv", na_values=['nan', '?'])
data['datetime'] = pd.to_datetime(data['timestamp'])
data.set_index('datetime', inplace=True)
data.drop('timestamp', axis=1, inplace=True)
data.fillna(0, inplace=True)
values = data.values.astype('float32')
print(data.head(4))

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)
scaled_df = pd.DataFrame(scaled_data)
print(scaled_df.head(4))

# Create time series data
def prepare_time_series(data, lag=5, target_col=3):
    temp_data = data.copy()
    temp_data["index"] = range(1, len(temp_data) + 1)
    temp_data = temp_data.iloc[:-lag, :]
    temp_data.set_index('index', inplace=True)

    target_data = data.copy()
    target_data = target_data.iloc[lag:, target_col]
    target_df = pd.DataFrame(target_data, columns=["target"])

    target_df["index"] = range(1, len(target_df) + 1)
    target_df.set_index('index', inplace=True)

    final_data = pd.concat([temp_data, target_df], axis=1)
    return final_data

# Prepare data for modeling
reframed_data = prepare_time_series(scaled_df, lag=1, target_col=3)
reframed_data.fillna(0, inplace=True)
reframed_data.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)']
print(reframed_data.head())

# Split data into training and testing sets
split_index = int(len(data) * 0.7)
train_data = reframed_data.values[:split_index, :]
test_data = reframed_data.values[split_index:, :]

train_X, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Define model architecture
def build_model(trial):
    lstm_units_1 = trial.suggest_int('lstm_units_1', 50, 150, step=25)
    gru_units_1 = trial.suggest_int('gru_units_1', 30, 100, step=20)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 30, 70, step=10)
    gru_units_2 = trial.suggest_int('gru_units_2', 20, 50, step=10)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])),
        Dropout(dropout_rate),
        GRU(gru_units_1, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, return_sequences=True),
        Dropout(dropout_rate),
        GRU(gru_units_2),
        Dropout(dropout_rate),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
    return model

# Define objective function
def objective(trial):
    model = build_model(trial)
    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=36,
        batch_size=32,
        verbose=0,
        shuffle=False
    )
    return history.history['val_loss'][-1]

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Display best parameters
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")

# Train final model with best parameters
best_params = study.best_trial.params
final_model = build_model(study.best_trial)
final_model.fit(train_X, train_y, epochs=36, validation_data=(test_X, test_y), batch_size=32, shuffle=False)

# Predict on test set
test_predictions = final_model.predict(test_X)

# Plot training history
plt.plot(final_model.history.history['loss'], label='Train Loss', color='brown')
plt.plot(final_model.history.history['val_loss'], label='Validation Loss', color='blue')
plt.legend()
plt.show()

# Evaluate predictions
mse = mean_squared_error(test_y, test_predictions)
r2 = r2_score(test_y, test_predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test_y, test_predictions)

print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Plot predictions vs actual values
plt.plot(test_y, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.legend()
plt.show()

# Save model as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("optimized_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully saved as TFLite!")
