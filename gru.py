import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.utils import plot_model

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

def create_ts_data(dataset, lookback=2016, predicted_col=4):
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

model_gru = Sequential()
model_gru.add(GRU(75, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
model_gru.add(GRU(units=30, return_sequences=True))
model_gru.add(GRU(units=30))
model_gru.add(Dense(units=1))

model_gru.compile(loss='mae', optimizer='adam')

lstm_history = model_gru.fit(train_X, train_y, epochs=10,validation_data=(test_X, test_y), batch_size=64, shuffle=False)
pred_y =  model_gru.predict(test_X)

pyplot.plot(lstm_history.history['loss'], label='gru train', color='brown')
pyplot.plot(lstm_history.history['val_loss'], label='gru test', color='blue')
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
fp = model_gru.predict(tra)
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

converter = tf.lite.TFLiteConverter.from_keras_model(model_gru)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("aisola_gru.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil disimpan dalam format TFLite!")

plt.figure(figsize=(10, 5))
plt.plot(lstm_history.history['loss'], label='Training Loss', color='brown')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('lstm_loss_plot.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_y[:500], label='Actual kWh', color='blue')
plt.plot(pred_y[:500], label='Predicted kWh', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('kWh')
plt.title('Actual vs Predicted (Zoomed to 500 samples)')
plt.legend()
plt.grid(True)
plt.savefig('lstm_actual_vs_predicted.png')
plt.show()

residuals = test_y - pred_y.flatten()
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel('Residuals')
plt.title('Residual Distribution')
plt.grid(True)
plt.savefig('lstm_residuals_distribution.png')
plt.show()

plt.figure(figsize=(10, 5))
epochs = range(1, len(lstm_history.history['loss']) + 1)
plt.bar(epochs, lstm_history.history['loss'], label='Training Loss', color='brown', alpha=0.6)
plt.bar(epochs, lstm_history.history['val_loss'], label='Validation Loss', color='blue', alpha=0.6)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Bar Chart')
plt.legend()
plt.grid(True)
plt.savefig('lstm_loss_barchart.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(test_y, pred_y, alpha=0.6, edgecolors='k', label='Predicted vs Actual')
plt.xlabel('Actual kWh')
plt.ylabel('Predicted kWh')
plt.title('Predicted vs Actual kWh (Scatter Plot)')
plt.grid(True)
plt.legend()
plt.savefig('lstm_scatter_plot.png')
plt.show()
