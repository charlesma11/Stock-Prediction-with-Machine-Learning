import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
filename = "AAPL_Corona3.csv"
df = pd.read_csv(filename)
print(df.info())
print(df.head())

df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='ignore')
df['Date'] = pd.to_numeric(df['Date'])
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

close_data = df['Close'].values
close_data = close_data.reshape(-1, 1)
sc = MinMaxScaler(feature_range = (0, 1))
close_data_scaled = sc.fit_transform(close_data)


split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data_scaled[:split]
close_test = close_data_scaled[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

look_back = 5

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, CuDNNLSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.25))

model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Dropout(0.25))

model.add(Bidirectional(LSTM(10)))
model.add(Dropout(0.25))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit_generator(train_generator, epochs=50, verbose=1)
prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))
trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "AAPL Stock",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

close_data_scaled = close_data_scaled.reshape((-1, 1))
def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]
    prediction_list = prediction_list.reshape(-1, 1)

    return prediction_list


def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates


num_prediction = 15
forecast = predict(num_prediction, model)
forecast = sc.inverse_transform(forecast)
forecast = forecast.reshape((-1))
forecast_dates = predict_dates(num_prediction)
trace1 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Prediction'
)
fig = go.Figure(data=[trace1], layout=layout)
fig.show()