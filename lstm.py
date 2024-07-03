import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = yf.download('NFTY', period='5y', interval='1d')
df.head()

df.tail()

df = df.reset_index()
df.head()

df = df.drop(['Date','Adj Close'],axis = 1)
df.head()

plt.plot(df.Close)

ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')

ma100 = df.Close.rolling(100).mean()
ma100

ma200 = df.Close.rolling(200).mean()
ma200

plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

df.shape

#splitting data into train and test

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

data_training.head()

data_testing.head()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array.shape

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100: i])
  y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape

#ML model

from keras. layers import Dense, Dropout, LSTM
from keras. models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
               input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 60, activation = 'relu', return_sequences = True,))
model.add(Dropout(0.3))


model.add(LSTM(units = 80, activation = 'relu', return_sequences = True,))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.summary()

model.compile(optimizer='adam',loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

# model.save('keras_model.h5')

past_100_day = data_training.tail(100)

final_df = pd.concat ([past_100_day,data_testing], ignore_index = True)
final_df.head()

input_data = scaler.fit_transform(final_df)
input_data

input_data.shape

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test , y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#Making Predictions

y_predicted = model.predict(x_test)

y_predicted.shape

y_test

y_predicted

scaler.scale_

scale_factor = 1/0.05953089
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


#future prediciton

# Extract the last 100 days of data
past_100_days = df.tail(100)

# Concatenate the past 100 days with future data for prediction
future_data = pd.concat([past_100_days, data_testing], ignore_index=True)

# Scale the input data
input_data = scaler.fit_transform(future_data)

# Prepare input data for prediction
x_future = []
for i in range(100, input_data.shape[0]):
    x_future.append(input_data[i-100:i])

# Convert to numpy array
x_future = np.array(x_future)

# Make predictions
predicted_prices = model.predict(x_future)

# Scale the predicted prices back to original scale
predicted_prices = predicted_prices * scale_factor

# Plot the original and predicted prices
fig3 = plt.figure(figsize=(12, 6))
plt.plot(data_testing.index, data_testing.values, label='Original Prices')
plt.plot(data_testing.index, predicted_prices, label='Predicted Prices', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted Future Prices')
plt.legend()
plt.show()

# Display the predicted prices
print("Predicted Future Prices:")
print(predicted_prices)

# Future Prediction
future_days = 10  # Number of days to predict into the future
x_input = input_data[-100:]  # Use the last 100 days as input for predicting the next 'future_days'

future_predictions = []  # List to store future predictions

for i in range(future_days):
    x_input = x_input.reshape((1, 100, 1))  # Reshape input for the model
    y_pred = model.predict(x_input)  # Predict next day's price
    future_predictions.append(y_pred[0, 0])  # Append the prediction to the list
    x_input = np.append(x_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)  # Update input data by shifting one step forward

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

for day, prediction in enumerate(future_predictions, 1):
    print(f"Day {day}: Predicted Price = {prediction}")
    

# Model Accuracy
from sklearn.metrics import mean_squared_error

# Inverse transform to get the original scale
y_test_original = y_test / scale_factor
y_predicted_original = y_predicted / scale_factor

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_original, y_predicted_original)
print('Mean Squared Error (MSE):', mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Calculate Mean Absolute Percentage Error (MAPE)

# Model Accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Inverse transform to get the original scale
y_test_original = y_test / scale_factor
y_predicted_original = y_predicted / scale_factor

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_original, y_predicted_original)
print('Mean Squared Error (MSE):', mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_original, y_predicted_original)
print('Mean Absolute Error (MAE):', mae)

# Calculate Mean Absolute Percentage Error (MAPE)
# MAPE can be calculated as follows
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE using the function defined above
mape = mean_absolute_percentage_error(y_test_original, y_predicted_original)
print('Mean Absolute Percentage Error (MAPE):', mape)
