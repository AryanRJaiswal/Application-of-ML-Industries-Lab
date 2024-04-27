import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
file_path = r"C:\Users\asus\Downloads\airline-passengers.csv"
df = pd.read_csv(file_path, index_col='Month', parse_dates=True)
df.index.freq = 'MS'
df.shape
df.columns
plt.figure(figsize=(20, 4))
plt.plot(df.Passengers, linewidth=2)
plt.show()
nobs = 12
df_train = df.iloc[:-nobs]
df_test = df.iloc[-nobs:]
df_train.shape
df_test.shape
scaler = MinMaxScaler()
scaler.fit(df_train)
scaled_train = scaler.transform(df_train)
scaled_test = scaler.transform(df_test)
n_inputs = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_inputs, batch_size=1)
for i in range(len(generator)):
    X, y = generator[i]
    print(f'\n {X.flatten()} and {y}')
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_inputs, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(generator, epochs=50)
plt.plot(model.history.history['loss'])
last_train_batch = scaled_train[-12:]
last_train_batch = last_train_batch.reshape(1, 12, 1)
last_train_batch
model.predict(last_train_batch)
# Forecasting
y_pred = []
first_batch = scaled_train[-n_inputs:]
current_batch = first_batch.reshape(1, n_inputs, n_features)
for i in range(len(scaled_test)):
    batch = current_batch
    pred = model.predict(batch)[0]
    y_pred.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
y_pred
scaled_test
df_test

y_pred_transformed = scaler.inverse_transform(y_pred)
y_pred_transformed = np.round(y_pred_transformed, 0)
y_pred_final = y_pred_transformed.astype(int)

y_pred_final
df_test.values, y_pred_final
df_test['Predictions'] = y_pred_final
df_test

plt.figure(figsize=(15, 6))
plt.plot(df_train.index, df_train.Passengers, linewidth=2, color='black', label='Train Values')
plt.plot(df_test.index, df_test.Passengers, linewidth=2, color='green', label='True Values')
plt.plot(df_test.index, df_test.Predictions, linewidth=2, color='red', label='Predicted Values')
plt.legend()
plt.show()

sqrt(mean_squared_error(df_test.Passengers, df_test.Predictions))
