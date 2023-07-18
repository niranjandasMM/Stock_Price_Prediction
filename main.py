import numpy as np
import matplotlib.pyplot as plt
from train import PolynomialRegression
from sklearn.metrics import r2_score 
from data import x_train, x_test, y_train, y_test

# np.random.seed(42)
# # X = np.random.rand(1000, 3)
# X = np.array( [[1,2,3], [4,5,6], [7,8,9]] )
# y = 5 * (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) + np.random.rand(3)
# y = y.reshape(-1, 1)

degree = [2]

model = PolynomialRegression(degrees=degree)
model.train(x_train, y_train, epochs=1500, lr=0.9)

y_test_pred = model.predict(x_test, test=True)
print(f"r2 score  is : {r2_score(y_test, y_test_pred)}")



