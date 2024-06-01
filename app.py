import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential as TfSequential
from tensorflow.keras.layers import LSTM as TfLSTM, Dense as TfDense
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator
from ta import momentum
from tensorflow.keras.models import load_model
from datetime import datetime
import streamlit as st
import time


# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1d')
    return hist



# Fetch historical stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create the input data for LSTM
def create_input_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Function to prepare the data with ADX indicator
def prepare_data_with_adx(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    adx_indicator = ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
    data['ADX'] = adx_indicator.adx()
    data.dropna(inplace=True)
    return data

# Function to generate additional features
def generate_features(dataset):
    dataset['HL'] = dataset['High'] - dataset['Low']
    dataset['HC'] = abs(dataset['High'] - dataset['Close'].shift(1))
    dataset['LC'] = abs(dataset['Low'] - dataset['Close'].shift(1))
    dataset['TR'] = dataset[['HL', 'HC', 'LC']].max(axis=1)
    dataset['ADL'] = (dataset['Close'] - dataset['Low']) / (dataset['High'] - dataset['Low']) * dataset['Volume']
    return dataset[['Close', 'Volume', 'ADL']]

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1):
        data_X.append(dataset[i:(i + look_back), 0])
        data_Y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_Y)

# Function to calculate Aroon indicator
def get_aroon(df, period=14):
    df['Up'] = df['Close'].rolling(window=period).apply(lambda x: len(x[x == x.max()]), raw=True)
    df['Down'] = df['Close'].rolling(window=period).apply(lambda x: len(x[x == x.min()]), raw=True)
    df['Aroon_Up'] = (period - df['Up']) / period * 100
    df['Aroon_Down'] = (period - df['Down']) / period * 100
    df.drop(['Up', 'Down'], axis=1, inplace=True)
    return df

# Function to create Bollinger Bands
def create_bollinger_bands(data, window_size=20):
    data['MA'] = data['Close'].rolling(window=window_size).mean()
    data['std_dev'] = data['Close'].rolling(window=window_size).std()
    data['Upper_band'] = data['MA'] + (data['std_dev'] * 2)
    data['Lower_band'] = data['MA'] - (data['std_dev'] * 2)
    return data

# Function to prepare data for LSTM
def prepare_data(data, window_size=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    x_data = []
    y_data = []
    for i in range(len(scaled_data) - window_size):
        x_data.append(scaled_data[i:i + window_size, 0])
        y_data.append(scaled_data[i + window_size, 0])
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data, scaler

# Function to calculate On-Balance Volume (OBV)
def calculate_obv(data):
    obv = []
    prev_obv = 0
    for i in range(len(data)):
        if i == 0:
            obv.append(0)
        elif data['Close'][i] > data['Close'][i - 1]:
            prev_obv += data['Volume'][i]
            obv.append(prev_obv)
        elif data['Close'][i] < data['Close'][i - 1]:
            prev_obv -= data['Volume'][i]
            obv.append(prev_obv)
        else:
            obv.append(prev_obv)
    return obv

# Function to calculate Stochastic Oscillator
def stochastic_oscillator(df, n=14):
    df['L14'] = df['Low'].rolling(window=n).min()
    df['H14'] = df['High'].rolling(window=n).max()
    df['%K'] = (df['Close'] - df['L14']) / (df['H14'] - df['L14']) * 100
    return df.drop(['L14', 'H14'], axis=1)










current_datetime = datetime.now()

st.title('Stock Prediction')
st.write('Today--->')
st.write(current_datetime.date())
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

if user_input:
    # Fetch the stock data
    data = get_stock_data(user_input)
    
    if not data.empty:
        # Display the current price
        current_price = data['Close'].iloc[-1]
        st.write(f"**Current price of {user_input}:** {current_price:.2f}")
        
        # Fetch historical data for chart
        hist = yf.Ticker(user_input).history(period="6mo")
        st.write("### Stock Price Chart")
        st.line_chart(hist['Close'])
    else:
        st.write("Invalid ticker or no data available.")




tickerData = yf.Ticker(user_input)
df = tickerData.history(period='1d', start='2021-01-01', end=current_datetime.date())

#Describe_data

st.subheader('Data from 2021 to Present')
st.write(df.describe())

#visualize

st.subheader('Closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Spliting data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
st.subheader('Training chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data_training.Close)
st.pyplot(fig)

data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
st.subheader('Testing chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data_testing.Close)
st.pyplot(fig)

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Loading model

model = load_model('keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)

data_testing_array = data_testing.values  # Convert to NumPy array

#Final graph

min_length = min(len(y_predicted), len(data_testing_array))
data_testing_trimmed = data_testing_array[-min_length:]
y_predicted_trimmed = y_predicted[-min_length:]

st.subheader('Prediction with normal LSTM')

fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_predicted_trimmed, 'r', label = 'Predicted')
plt.plot(data_testing_trimmed, 'k', linewidth=2, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

closing_price_lstm = y_predicted[y_predicted.shape[0]-1][0]












#ADDITIONAL ALGORITHMS

ticker = user_input
start_date = '2020-01-01'

current_datetime = datetime.now()
end_date=current_datetime.date()
window_size = 30  # Adjust window size as per your preference

# Download data and prepare features with ADX indicator
data_with_adx = prepare_data_with_adx(ticker, start_date, end_date)

# Preprocess the data
scaled_data, scaler = preprocess_data(data_with_adx)

# Create input data for LSTM
X, y = create_input_data(scaled_data, window_size)

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Reshape the input data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Load the model
loaded_model = load_model('model_adi.h5')

# Use the loaded model to make predictions
predicted_data = loaded_model.predict(X_test)
predicted_data_adx = scaler.inverse_transform(predicted_data)
closing_price_adx = predicted_data_adx[-1][0]

min_length = min(len(predicted_data_adx), len(data_testing_array))
data_testing_trimmed = data_testing_array[-min_length:]
predicted_data_adx_trimmed = predicted_data_adx[-min_length:]


st.subheader('Prediction with ADX')
fig3 = plt.figure(figsize = (12, 6))
plt.plot(predicted_data_adx_trimmed, 'r', label = 'Predicted with ADX')
plt.plot(data_testing_trimmed, 'k', linewidth=2, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)


data = yf.download(ticker, start="2020-01-01", end=current_datetime.date())
data = get_aroon(data)
dataset = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 14
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Load the model
loaded_model = load_model('model_aroon.h5')

trainPredict = loaded_model.predict(trainX)
testPredict = loaded_model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)

min_length = min(len(testPredict), len(data_testing_array))
data_testing_trimmed = data_testing_array[-min_length:]
testPredict_trimmed = testPredict[-min_length:]

st.subheader('Prediction with Aroon')
fig3 = plt.figure(figsize = (12, 6))
plt.plot(testPredict_trimmed, 'r', label = 'Predicted with Aroon')
plt.plot(data_testing_trimmed, 'k', linewidth=2, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
    
    # Predicting next closing price
last_data = dataset[-look_back:]
last_data = np.reshape(last_data, (1, 1, look_back))
next_closing_price = loaded_model.predict(last_data)

next_closing_price = scaler.inverse_transform(next_closing_price.reshape(-1, 1))
closing_price_aroon = next_closing_price[0][0]



stock_data = fetch_stock_data(ticker, start_date, end_date)

# Calculate Bollinger Bands
stock_data = create_bollinger_bands(stock_data)

# Prepare data for LSTM
data = stock_data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define window size for LSTM
window_size = 20

# Create dataset for LSTM
X, Y = create_dataset(scaled_data, window_size)

# Reshape data for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Load the model
loaded_model = load_model('model_bb.h5')

last_window = scaled_data[-window_size:]
last_window = np.reshape(last_window, (1, window_size, 1))
predicted_closing_price = loaded_model.predict(last_window)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

closing_price_bb = predicted_closing_price[0][0]

df = fetch_stock_data(ticker, start_date, end_date)

# Calculate Moving Average
df['100_MA'] = df['Close'].rolling(window=100).mean()

# Drop NaN values
df.dropna(inplace=True)

# Prepare data
window_size = 100
x_data, y_data, scaler = prepare_data(df['Close'], window_size)
# Train-test split
train_size = int(len(x_data) * 0.8)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]


# Load the model
loaded_model = load_model('model_ma100.h5')
# Predictions
predictions = loaded_model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

st.subheader('Closing price vs time chart w 100MA, 200MA')
ma200 = df.Close.rolling(200).mean()
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, 'g')
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

min_length = min(len(predictions), len(data_testing_array))
data_testing_trimmed = data_testing_array[-min_length:]
predictions_trimmed = predictions[-min_length:]

st.subheader('Prediction with Moving Average 100 days')
fig3 = plt.figure(figsize = (12, 6))
plt.plot(predictions_trimmed, 'r', label = 'Predicted with MA100')
plt.plot(data_testing_trimmed, 'k', linewidth=2, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Predict next day's closing price
last_window = df['Close'].tail(window_size).values.reshape(-1, 1)
last_window_scaled = scaler.transform(last_window)
last_window_scaled = np.reshape(last_window_scaled, (1, window_size, 1))
next_day_prediction_scaled = loaded_model.predict(last_window_scaled)
next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)

closing_price_ma100 = next_day_prediction[0][0]



# Fetch historical data
stock_data = fetch_stock_data(ticker, start_date, end_date)
# Calculate Stochastic Oscillator
stock_data = stochastic_oscillator(stock_data)
# Preprocess data
scaled_data, scaler = preprocess_data(stock_data)
# Prepare data for LSTM
X, y = [], []
for i in range(len(scaled_data) - 60):
    X.append(scaled_data[i:i+60, 0])
    y.append(scaled_data[i+60, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load the model
loaded_model = load_model('model_so.h5')

predictions_so = loaded_model.predict(X_test)
predictions_so = scaler.inverse_transform(predictions_so)
closing_price_so = predictions_so[-1][0]

min_length = min(len(predictions_so), len(data_testing_array))
data_testing_trimmed = data_testing_array[-min_length:]
predictions_so_trimmed = predictions_so[-min_length:]


st.subheader('Prediction with Stochastic Oscillator')
fig3 = plt.figure(figsize = (12, 6))
plt.plot(predictions_so_trimmed, 'r', label = 'Predicted with SO')
plt.plot(data_testing_trimmed, 'k', linewidth=2, label='Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)




# Find the minimum length among all the datasets
min_length = min(len(predictions_so), len(predictions), len(testPredict), len(predicted_data_adx), len(y_predicted), len(data_testing_array))

# Trim each dataset to the minimum length
data_testing_trimmed = data_testing_array[-min_length:]
predictions_so_trimmed = predictions_so[-min_length:]
predictions_trimmed = predictions[-min_length:]
testPredict_trimmed = testPredict[-min_length:]
predicted_data_adx_trimmed = predicted_data_adx[-min_length:]
y_predicted_trimmed = y_predicted[-min_length:]

# Plotting
st.subheader('Testing Chart and All Predictions')
fig = plt.figure(figsize=(15, 8))  # Reduced the figure size to make it more manageable

# Plotting the actual data as an area chart for better visibility
plt.plot(data_testing_trimmed, 'k-', linewidth=2, label='Actual')

# Plotting predictions with enhanced visibility
plt.plot(predictions_so_trimmed, 'r-', linewidth=2, label='Predicted with SO')
plt.plot(predictions_trimmed, 'b--', linewidth=2, label='Predicted with MA100')
plt.plot(testPredict_trimmed, 'g-.', linewidth=2, label='Predicted with Aroon')
plt.plot(predicted_data_adx_trimmed, 'c:', linewidth=2, label='Predicted with ADX')
plt.plot(y_predicted_trimmed, 'm-', linewidth=2, label='Predicted with LSTM')

# Adding grid, title, and labels
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Testing Chart and All Predictions', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Price', fontsize=14)

# Enhancing the legend
plt.legend(loc='best', fontsize=12)

# Display the plot
st.pyplot(fig)


# Calculate the average closing price
average_closing_price = (closing_price_lstm + closing_price_adx + closing_price_aroon + closing_price_bb + closing_price_ma100 + closing_price_so) / 6

# Display individual closing prices
st.write("### Closing Prices from Different Models:")
st.write(f"LSTM: {closing_price_lstm:.2f}")
st.write(f"ADX: {closing_price_adx:.2f}")
st.write(f"Aroon: {closing_price_aroon:.2f}")
st.write(f"Bollinger Bands: {closing_price_bb:.2f}")
st.write(f"MA100: {closing_price_ma100:.2f}")
st.write(f"Stochastic Oscillator: {closing_price_so:.2f}")

# Display the average closing price in a bigger font
# Calculate percentage change
percentage_change = ((average_closing_price-current_price) / current_price) * 100

# Determine the color based on the percentage change
percentage_color = "red" if percentage_change < 0 else "green"

# Display average closing price
st.markdown(f"## **Average Closing Price Predicted: {average_closing_price:.2f}**")

# Display current price
st.markdown(f"## **Current Price: {current_price:.2f}**")

# Display percentage change with color
st.markdown(f"## **Percentage Change Predicted: <span style='color:{percentage_color}'>{percentage_change:.2f}%</span>**", unsafe_allow_html=True)