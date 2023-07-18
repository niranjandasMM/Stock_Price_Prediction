import numpy as np
import pandas as pd
import sys

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class PolynomialRegression:
    def __init__(self, degrees):
        self.degrees = degrees
        self.w = 0
        self.b = 0

    def gradients(self, X, X_transformed,  y_train, y_pred, lr):
        m = X.shape[0]  ##(1000,3)
        error = y_pred - y_train
        # print(f" shape is : {X.shape, self.w.shape, error.shape}") ## ((3, 3), (6, 1), (3, 1))
        
        dw = (1/m) * np.dot(X_transformed.T, error)
        db = (1/m) * np.sum(error)

        self.w -= lr * dw
        self.b -= lr * db

    def predict(self, X, test=False):
        if test:
            X_transformed = self.x_transform(X)
            return np.dot(X_transformed, self.w) + self.b
        else:
            return np.dot(X, self.w) + self.b

    def train(self, X_train, y_train, epochs, lr):
        X_transformed = self.x_transform(X_train)
        m, n = X_transformed.shape  ## no. of samples, no. of features

        self.w = np.zeros((n, 1))  
        self.b = 0
        losses = []
        
        for epoch in range(epochs):
            # y_pred = w1*x1 + w2*x1^2 + w3*x2 + w4*x2^2 + w5*x3 + w6*x3^2 + b ## degree = 2
            y_pred = self.predict(X_transformed)
            self.gradients(X_train, X_transformed, y_train, y_pred, lr)
            # print(f"self.w and self.b are : {self.w, self.b}")

            loss = mean_squared_error(y_train, y_pred) 
            
            if epoch % 100 == 0:
                # Initialize an empty list to store the data for the maximum difference
                data = []

                # Iterate over the predictions and actual values
                for x, y in zip(y_pred, y_train):
                    diff = abs(x - y)  # Calculate the absolute difference
                    data.append({"y_predicted": x, "actual": y, "difference": diff})

                # Create the DataFrame outside the loop
                df = pd.DataFrame(data)
                # print(df)
                max_diff = df['difference'].max()
                
                sys.stdout.write(
                    "\n" +
                    "I:" + str(epoch) +
                    " Train-Err:" + str(loss / float(len(X_train)))[0:5] +
                    " MAX Difference: " + str(max_diff) +
                    "\n"
                )
        print(f" training r2 score is : {r2_score(y_train, y_pred)}")

    def x_transform(self, X):
        t = X.copy()
        for i in self.degrees:
            X_transformed = np.append(X, t**i, axis=1)
        return X_transformed

