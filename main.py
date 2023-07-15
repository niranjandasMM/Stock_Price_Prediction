import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time 
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
from train import MultipleLinearRegression
from data import get_data_and_preprocess
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation
from streamlit import components

print(f"Hey app started....")
st.markdown("<div align='center'><h1 style='font-weight: bold'> Stock Price Prediction App</h1></div>", unsafe_allow_html=True)


# Define the content
content = """
<span style='color:purple'>This app predicts the <b>Stock Price</b> of a company using the <b>Multiple Linear Regression</b>!
The data is fetched from </span> <span style='color:blue'><b>Yahoo Finance</b></span> <span style='color:purple'> and then the model is trained for the first <b>4.25 hours</b> of data.
The model is trained on the data from <b>9:15 AM to 1:30 PM</b> and tested on the data from <b>1:30 PM to 3:30 PM</b>.
The model primarily predicts for day trading.
NOTE: It can't forecast future prices, this model can only predict prices based on historical data. </span>
"""

# Show an expander to toggle visibility
with st.expander("About this MODEL -MLR"):
    st.markdown(content, unsafe_allow_html=True)

# Apply CSS styling using Markdown
st.markdown(
    """
    <style>
    span {
        font-weight: bold;
        font-size: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Calculate the date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=7)

# Create the list of selectable dates, excluding Saturdays and Sundays
date_range = []
current_date = start_date
while current_date <= end_date:
    if current_date.weekday() < 5:  # Monday to Friday (0 to 4)
        date_range.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

# Create select boxes for start date and end date
start_date = st.selectbox('Select Start Date', date_range)
end_date = st.selectbox('Select End Date', date_range)

# Display the selected dates
st.write("Selected Start Date:", start_date)
st.write("Selected End Date:", end_date)

if st.button('Submit'):
    X_train, y_train, X_test, y_test = get_data_and_preprocess(stock_symbol='BANKBARODA.NS',
                                                            start_date=start_date, 
                                                            end_date=end_date)

    model = MultipleLinearRegression()
    y_hat = model.train(X_train, y_train, epochs=350)

    y_pred = model.predict(X_test, test=True)
    r2 = r2_score(y_test, y_pred)
    st.write("Test R2 Score:", r2)


    st.title('Training (9:15AM -1:30PM) and Testing(1:30PM - 3:30PM)  Predicted vs Actual Stock Prices')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(X_train['Datetime'], y_hat, label='Predicted (Train)', color='green')
    ax.plot(X_train['Datetime'], y_train, label='Actual (Train)', color='darkblue')
    ax.plot(X_test['Datetime'], y_pred, label='Predicted (Test)', color='green')
    ax.plot(X_test['Datetime'], y_test, label='Actual (Test)', color='red')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Stock Price')
    ax.set_title('Predicted vs Actual Stock Prices')
    ax.legend()
    st.pyplot(fig)

