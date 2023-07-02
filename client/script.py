import requests
import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax

filename = "/usr/data/model.txt"

class MCLogisticRegressor:
    def __init__(self) -> None:
        pass

    def train(self, X, Y) -> None:
        self.encoding_map = dict.fromkeys(np.unique(Y))
        for k, ele in enumerate(self.encoding_map.keys()):
            self.encoding_map[ele] = k

        self.decoded_map = {}
        for k, ele in enumerate(self.encoding_map.keys()):
            self.decoded_map[k] = ele
        
        self.W = self.__gradient_descent__(X, Y)

    def __gradient_descent__(self, X, Y, max_iterations=1000, alpha=0.1, mu=0.01):
        y_onehot_encoded = self.__onehot_encoded__(Y)
        W = np.zeros((X.shape[1], y_onehot_encoded.shape[1]))

        for i in range(max_iterations):
            W -= alpha * self.__gradient__(X, y_onehot_encoded, W, mu)
        
        return W

    def __gradient__(self, X, Y, W, mu):
        Z = X @ W
        P = softmax(Z, axis=1)
        grad = (-1 / X.shape[0]) * (X.T @ (Y - P)) + 2 * mu * W
        return grad        

    def __loss__(self, X, Y, W):
        Z = X @ W
        loss = (1 / X.shape[0]) * (-np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def __onehot_encoded__(self, y):
        y_onehot_encoded = np.zeros((y.shape[0], len(self.encoding_map.keys())))
        for i, ele in enumerate(y_onehot_encoded):
            ele[self.encoding_map[y[i]]] = 1
        
        return y_onehot_encoded

def trainAndSaveModel():
    csv_data = pd.read_csv("/usr/data/iris.data")
    X = csv_data.iloc[:, :-1].values
    Y = csv_data.iloc[:, -1].values

    X_train, _, Y_train, __ = train_test_split(X, Y, test_size=0.4, random_state=42)
    logreg = MCLogisticRegressor()
    logreg.train(X_train, Y_train)

    with open(filename, 'w') as f:
        np.savetxt(f, logreg.W, fmt='%.3f')

def client():
    url = 'http://server:8000/server'  # Replace with the server URL
    message = ""
    
    if os.path.exists(filename):
        file = open(filename, "r")
        message = file.read()
        file.close()
    else:
        print("Failed")
    
    response = requests.post(url, {"message": message})
    if response.status_code == 200:
        print('Server response:', response.text)
    else:
        print('Error:', response.status_code)

if __name__ == '__main__':
    trainAndSaveModel()
    client()
