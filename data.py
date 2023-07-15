import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def get_data_and_preprocess(stock_symbol, start_date, end_date):
    # Define the stock symbol and the date
    stock_symbol = stock_symbol
    start_date = start_date
    end_date = end_date

    # Fetch the intraday stock data using yfinance
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval='1m')

    # Extract the features (Open, High, Low, Volume, Datetime)
    features = stock_data[['Open', 'High', 'Low', 'Volume']]

    # Extract the target variable (Close)
    target = stock_data['Close']
    ##################### Fetching data is done ############################
    # As the Data have Daetime as index, we are resetting the index and dropping the Datetime column
    features = features.reset_index()
    target = target.reset_index()
    target = target.drop('Datetime', axis=1)

    # # Split the data into training and testing sets based on datetime condition
    X_train = features[features['Datetime'] <= pd.Timestamp(f'{start_date} 13:30:00+05:30')]
    y_train = pd.DataFrame(target.loc[features[features['Datetime'] <= pd.Timestamp(f'{start_date} 13:30:00+05:30')].index, 'Close'])

    X_test = features[features['Datetime'] > pd.Timestamp(f'{start_date} 13:30:00+05:30')]
    y_test = pd.DataFrame(target.loc[features[features['Datetime'] > pd.Timestamp(f'{start_date} 13:30:00+05:30')].index, 'Close'])
    
    # Select the columns to scale (excluding 'Datetime')
    columns_to_scale = ['Open', 'High', 'Low', 'Volume']

    # Create the StandardScaler object
    scaler = StandardScaler()

    # Apply feature scaling to the selected columns in X_train
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

    # Apply feature scaling to the selected columns in X_test
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train, y_train, X_test, y_test

