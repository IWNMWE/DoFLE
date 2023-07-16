import requests
import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax
import tensorflow as tf

#ilename = "/usr/data/model.txt"
hostname = os.environ['HOSTNAME']

def create_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

##class MCLogisticRegressor:
##    def __init__(self) -> None:
##        pass
##
##    def train(self, X, Y) -> None:
##        self.encoding_map = dict.fromkeys(np.unique(Y))
##        for k, ele in enumerate(self.encoding_map.keys()):
##            self.encoding_map[ele] = k
##
##        self.decoded_map = {}
##        for k, ele in enumerate(self.encoding_map.keys()):
##            self.decoded_map[k] = ele
##        
##        self.W = self.__gradient_descent__(X, Y)
##
##    def __gradient_descent__(self, X, Y, max_iterations=1000, alpha=0.1, mu=0.01):
##        y_onehot_encoded = self.__onehot_encoded__(Y)
##        W = np.zeros((X.shape[1], y_onehot_encoded.shape[1]))
##
##        for i in range(max_iterations):
##            W -= alpha * self.__gradient__(X, y_onehot_encoded, W, mu)
##        
##        return W

##   def __gradient__(self, X, Y, W, mu):
##       Z = X @ W
##       P = softmax(Z, axis=1)
##       grad = (-1 / X.shape[0]) * (X.T @ (Y - P)) + 2 * mu * W
##       return grad       
##   def __loss__(self, X, Y, W):
##       Z = X @ W
##       loss = (1 / X.shape[0]) * (-np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
##       return loss
##   def __onehot_encoded__(self, y):
##       y_onehot_encoded = np.zeros((y.shape[0], len(self.encoding_map.keys())))
##       for i, ele in enumerate(y_onehot_encoded):
##           ele[self.encoding_map[y[i]]] = 1
##       
##       return y_onehot_encoded
def trainAndSaveModel():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    x_train = x_train.reshape(-1,28*28) / 255.0
    x_test = x_test.reshape(-1,28*28) / 255.0
    model = create_model()
    model.fit(x_train,y_train,epochs = 1,validation_data = (x_test,y_test))
    return model


    

def client():
    url = 'http://server:8000/server'  # Replace with the server URL
    model = trainAndSaveModel()
    w = model.get_weights()
    for i in range(0,len(w)):
          w[i] =  w[i].tolist()
    response = requests.post(url, json = {"weights" : w,"config" : model.get_config()})


    if response.status_code == 200:
        print('Server response:', response.text)
    else:
        print('Error:', response.status_code)

if __name__ == '__main__':
    trainAndSaveModel()
    client()
