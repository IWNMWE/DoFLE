from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax
import tensorflow as tf
import docker

##class MCLogisticRegressor:
##    def __init__(self, X, Y, W) -> None:
##        self.W = W
##
##        self.encoding_map = dict.fromkeys(np.unique(Y))
##        for k, ele in enumerate(self.encoding_map.keys()):
##            self.encoding_map[ele] = k
##            
##        self.decoded_map = {}
##        for k, ele in enumerate(self.encoding_map.keys()):
##            self.decoded_map[k] = ele
##    
##    def predict(self, X):
##        Z = X @ self.W
##        P = softmax(Z, axis=1)
##        encoded_predictions = np.argmax(P, axis=1)
##        return np.array([self.decoded_map[x] for x in encoded_predictions])

app = Flask(__name__)

##def predict(w):
##    csv_data = pd.read_csv("/usr/data/iris.data")
##    X = csv_data.iloc[:, :-1].values
##    Y = csv_data.iloc[:, -1].values
##
##    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
##
##    logreg = MCLogisticRegressor(X_train, Y_train, w)
##    predictions = logreg.predict(X_test)
##    return np.sum(predictions == Y_test) / Y_test.shape[0] * 100

def getClientContainers():
    return docker.from_env().containers.list(filters={"name": "client"})

@app.route('/server', methods=['POST'])
def server():
    if request.method == 'POST':
        config = request.json["config"]
        model = tf.keras.Sequential().from_config(config)
        weights = request.json["weights"]
        for i in range(0,len(weights)):
            weights[i] =  np.array(weights[i])
        
        model.set_weights(weights)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)
        x_train = x_train.reshape(-1,28*28) / 255.0
        x_test = x_test.reshape(-1,28*28) / 255.0
        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return "evaluated " + str(model.evaluate(x_test,y_test))
        
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)