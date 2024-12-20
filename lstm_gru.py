import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
import tensorflow as tf
import optuna
from datetime import datetime


dataset = read_csv("data.csv", na_values=['nan', '?'])
dataset['dt'] = pd.to_datetime(dataset['timestamp'])
dataset.set_index('dt', inplace=True)
dataset.drop('timestamp', axis=1, inplace=True)
dataset.fillna(0, inplace=True)
values = dataset.values
values = values.astype('float32')
print(dataset.head(4))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
scaled = pd.DataFrame(scaled)
print(scaled.head(4))

def create_ts_data(dataset, lookback=5, predicted_col=3):
    temp = dataset.copy()
    temp["id"]= range(1, len(temp)+1)
    temp = temp.iloc[:-lookback, :]
    temp.set_index('id', inplace =True)
    
    predicted_value=dataset.copy()
    predicted_value = predicted_value.iloc[lookback:,predicted_col]
    predicted_value.columns=["Predcited"]
    predicted_value= pd.DataFrame(predicted_value)
    
    predicted_value["id"]= range(1, len(predicted_value)+1)
    predicted_value.set_index('id', inplace =True)
    final_df= pd.concat([temp, predicted_value], axis=1)
    #final_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)', 'var6(t-1)', 'var7(t-1)', 'var8(t-1)','var1(t)']
    #final_df.set_index('Date', inplace=True)
    return final_df

reframed_df= create_ts_data(scaled, 1,3)
reframed_df.fillna(0, inplace=True)

reframed_df.columns = ['var1(t-1)', 'var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)',]
print(reframed_df)

column_means = reframed_df.mean()
print(column_means)

values = reframed_df.values
training_sample =int( len(dataset) *0.7)
train = values[:training_sample, :]
test = values[training_sample:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def create_combined_model(trial):
    units_lstm1 = trial.suggest_int('units_lstm1', 50, 150, step=25)
    units_gru1 = trial.suggest_int('units_gru1', 30, 100, step=20)
    units_lstm2 = trial.suggest_int('units_lstm2', 30, 70, step=10)
    units_gru2 = trial.suggest_int('units_gru2', 20, 50, step=10)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    model = Sequential()
    model.add(LSTM(units_lstm1, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units_gru1, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units_lstm2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units_gru2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model

def objective(trial):
    model = create_combined_model(trial)
    
    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=36,
        batch_size=32,
        verbose=0,
        shuffle=False
    )
    
    val_loss = history.history['val_loss'][-1]
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best trial: {study.best_trial.params}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")
    
best_params = study.best_trial.params
    
model_combined = Sequential()
model_combined.add(LSTM(best_params['units_lstm1'], return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model_combined.add(Dropout(best_params['dropout_rate']))
model_combined.add(GRU(best_params['units_gru1'], return_sequences=True))
model_combined.add(Dropout(best_params['dropout_rate']))
model_combined.add(LSTM(best_params['units_lstm2'], return_sequences=True))
model_combined.add(Dropout(best_params['dropout_rate']))
model_combined.add(GRU(best_params['units_gru2']))
model_combined.add(Dropout(best_params['dropout_rate']))
model_combined.add(Dense(units=1))

optimizer = Adam(learning_rate=best_params['learning_rate'])
model_combined.compile(optimizer=optimizer, loss='mae')

lstm_history = model_combined.fit(train_X, train_y, epochs=36, validation_data=(test_X, test_y), batch_size=32, shuffle=False)
pred_y =  model_combined.predict(test_X)

pyplot.plot(lstm_history.history['loss'], label='lstm train', color='brown')
pyplot.plot(lstm_history.history['val_loss'], label='lstm test', color='blue')
pyplot.legend()
pyplot.show()

test_y.reshape(2591,1)
print(pred_y)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,5)

from sklearn.metrics import *
from math import sqrt

MSE = mean_squared_error(test_y, pred_y)
R2 = r2_score(test_y, pred_y)
RMSE = sqrt(mean_squared_error(test_y, pred_y))
MAE = mean_absolute_error(test_y, pred_y)

print(MSE)
print(R2)
print(RMSE)
print(MAE)

plt.plot(test_y, label = 'Actual')
plt.plot(pred_y, label = 'Predicted')
plt.legend()
plt.show()

tra = np.concatenate([train_X,test_X])
tes = np.concatenate([train_y,test_y])
fp = model_combined.predict(tra)
plt.plot(tes, label = 'Actual')
plt.plot(fp, label = 'Predicted')
plt.legend()
plt.show()

plt.plot(tes[:2000], label = 'Actual')
plt.plot(fp[:2000], label = 'Predicted')
plt.legend()
plt.show()

plt.plot(tes[:500], label = 'Actual')
plt.plot(fp[:500], label = 'Predicted')
plt.legend()
plt.show()

converter = tf.lite.TFLiteConverter.from_keras_model(model_combined)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("aisola_lstm_gru.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil disimpan dalam format TFLite!")