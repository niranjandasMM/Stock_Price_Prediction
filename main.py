import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time 

from train import MultipleLinearRegression
from data import get_data_and_preprocess
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation
import plotly.express as px
import plotly.graph_objects as go


X_train, y_train, X_test, y_test = get_data_and_preprocess(stock_symbol='BANKBARODA.NS',
                                                           start_date='2023-07-12', 
                                                           end_date='2023-07-13')

model = MultipleLinearRegression()
y_hat = model.train(X_train, y_train, epochs=350)

y_pred = model.predict(X_test, test=True)
r2 = r2_score(y_test, y_pred)
st.write("Test R2 Score:", r2)


st.title('Training (9:15AM -1:30PM) and Testing(1:30PM - 3:30PM)  Predicted vs Actual Stock Prices')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_train['Datetime'], y_hat, label='Predicted (Train)', color='green')
ax.plot(X_train['Datetime'], y_train, label='Actual (Train)', color='darkblue')
ax.plot(X_test['Datetime'], y_pred, label='Predicted (Test)', color='green')
ax.plot(X_test['Datetime'], y_test, label='Actual (Test)', color='red')
ax.set_xlabel('Datetime')
ax.set_ylabel('Stock Price')
ax.set_title('Predicted vs Actual Stock Prices')
ax.legend()
st.pyplot(fig)

