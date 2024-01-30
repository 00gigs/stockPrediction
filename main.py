import math
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




stock_data1 = yf.download('INTC', start='2023-01-01', end='2024-01-01')





# stock_data.head()



def plot1(stock_data1):
    train_predictions = model.predict(x_train)
    train_predictions = scaler.inverse_transform(train_predictions)
    plt.figure(figsize=(15, 8))
    plt.title('Training Data Predictions vs Actual')
    plt.plot(range(60, training_data_len), train_predictions, label='Training Predictions')
    plt.plot(range(60, training_data_len), values[60:training_data_len], label='Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,8))
    plt.title('Ticker chart')
    plt.plot(range(training_data_len, len(values)), prediction, label='Test Predictions')
    plt.plot(range(training_data_len, len(values)), values[training_data_len:], label='Actual Price[]')
    plt.xlabel('date')
    plt.ylabel('price ($)')
    plt.legend()
    plt.show()

#processing data for tensoflow with numpy
close_price = stock_data1['Close']
values = close_price.values
training_data_len = math.ceil(len(values)* .8)#first 80% of dataset

scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(values.reshape(-1,1))
trainData = scaledData[0: training_data_len , :]

x_train=[]
y_train=[]

for i in range(60,training_data_len):
    x_train.append(trainData[i-60:i,0])
    y_train.append(trainData[i,0])

x_train,y_train = np.array(x_train) , np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#preparation of test set 

testData = scaledData[training_data_len-60:,:] #80-60=last 20% of data set 

x_test = []
y_test = values[training_data_len:]

for i in range(60,len(testData)):
    x_test.append(testData[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Setting Up LSTM(long short term memory) Network Architecture
model = tf.keras.Sequential()

model.add(layers.LSTM(160, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(layers.LSTM(160,return_sequences=False))

model.add(layers.Dense(22))

model.add(layers.Dense(1))

model.add(layers.Dropout(0.5))

#optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
#Training LSTM Model
model.compile(optimizer='Adam', loss='mean_squared_error')

model.fit(x_train,y_train, batch_size=1,epochs=4)

#Model Evaluation / root mean square error (RMSE) metric to examine the performance of the model.

prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)


actual_values = np.array([value for value in y_test if not np.isnan(value)])


rsme = np.sqrt(np.mean((prediction - actual_values)**2))
mae = np.mean(np.abs(prediction - actual_values))
mape = np.mean(np.abs((actual_values - prediction) / actual_values)) * 100

print("RMSE:", rsme)
print("MAE:", mae)
print("MAPE:", mape)


