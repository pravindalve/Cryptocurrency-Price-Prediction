from datetime import date, timedelta
from binance.client import Client
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
client = Client("api_key", "api_secret", {"verify": False, "timeout": 20})

t = date.today().strftime("%d %b, %Y")
print(t)
m  = date.today() + timedelta(-365*2)
print(m.strftime("%d %b, %Y"))
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, str(m), str(t))

print(klines)
print(len(klines))

# Check if sufficient data is available
if len(klines) < 731:
  print("The sufficient data is not available for this pair")
  exit()

print(klines)
klines = np.array(klines)
klines[:,4]

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

closedf = pd.DataFrame({'Close':klines[:,4]})
print(closedf.shape)
closedf['Close'] = pd.to_numeric(closedf['Close'])
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

# we keep the training set as 60% and 40% testing set

training_size=int(len(closedf)*0.60)
test_size=len(closedf)-training_size
print(training_size)
print(test_size)
train_data,test_data=closedf[0:training_size],closedf[training_size:len(closedf)]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step)]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)
print(X_train)
print(y_train)
print(closedf)
# # reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print(X_train)
print(X_test)
model=Sequential()

model.add(LSTM(10,input_shape=(None,1),activation="relu"))

model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam")

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=150,batch_size=32,verbose=1)

### Lets Do the prediction and check performance metrics

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
print(train_predict.shape)
print(test_predict.shape)

train_predict1 = scaler.inverse_transform(train_predict)
test_predict2 = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
# Evaluation metrices RMSE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))

print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))


# predictedPrices = list(np.ndarray.flatten(train_predict)) + list(np.ndarray.flatten(test_predict))
# predictdf = pd.DataFrame(list(zip(maindf['Date'].tolist(), predictedPrices)), columns=["Date", "Close"])
# original = maindf[["Date", "Close"]]
# print(predictdf)
# fig = px.line(predictdf, x=predictdf.Date, y=predictdf.Close,labels={'date':'Date','close':'Close Stock'})

# fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
# fig.update_layout(title_text='predicted Whole period of timeframe of Bitcoin close price 2014-2022', plot_bgcolor='white',
#                   font_size=15, font_color='black')
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False)
# fig.show()

from sklearn.metrics import r2_score
R2=r2_score(y_test, test_predict)
print(R2)

