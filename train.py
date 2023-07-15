import sys
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def mse(y_pred, label):
    return np.mean((y_pred-label)**2)

class MultipleLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
    
    def optimize(self, x,  y_pred, y_train, learning_rate):
        error = [ (y_train[i] - y_pred[i] ) for i in range(len(x)) ]

        for i in range(len(x)):
            self.b1 += learning_rate * x[i][0] * error[i]
            self.b2 += learning_rate * x[i][1] * error[i]
            self.b3 += learning_rate * x[i][2] * error[i]
            self.b4 += learning_rate * x[i][3] * error[i]
            self.b0 += learning_rate * error[i]

    def predict(self,x, test=False):
        if test:
            x = x[['Open', 'High', 'Low', 'Volume']]
            x = np.array(x)
            y_pred = [self.b0 + ( (self.b1 * xi[0]) + (self.b2 * xi[1]) + (self.b3 * xi[2]) + (self.b4 * xi[3]) ) for xi in x]
            return y_pred
        else:
            y_pred = [self.b0 + ( (self.b1 * xi[0]) + (self.b2 * xi[1]) + (self.b3 * xi[2]) + (self.b4 * xi[3]) ) for xi in x]
            return y_pred
        
    def train(self, X_train, y_train, epochs):
        x_train = X_train[['Open', 'High', 'Low', 'Volume']] 
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        loss_list = []

        for epoch in range(epochs):
            y_pred = self.predict(x_train)
            loss = mse(y_pred, y_train)
            loss_list.append(loss)

            self.optimize(x_train, y_pred, y_train, learning_rate=0.001)

            # if epoch % 100 == 0:
            #     st.write(f"Epochs: {epoch} Train-Err: {loss / float(len(x_train)):.5f}")
            #     # sys.stdout.write(
            #     #     "\n"
            #     #     + "I:" + str(epoch)
            #     #     + " Train-Err:" + str(loss / float(len(x_train)))[0:5]
            #     #     + "\n"
            #     # )

        # Initialize an empty list to store the data
        data = []

        for x, y in zip(y_pred, y_train):
            diff = abs(x - y) 
            data.append({"y_pred": x, "y_train": y, "difference": diff})

        df = pd.DataFrame(data)
        max_diff = df['difference'].max()
        st.write("MAX Difference:", max_diff)

        r2 = r2_score(y_train, y_pred)
        st.write(" Training R2 Score:", r2)

        return y_pred