from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax

class MCLogisticRegressor:
    def __init__(self, X, Y, W) -> None:
        self.W = W

        self.encoding_map = dict.fromkeys(np.unique(Y))
        for k, ele in enumerate(self.encoding_map.keys()):
            self.encoding_map[ele] = k
            
        self.decoded_map = {}
        for k, ele in enumerate(self.encoding_map.keys()):
            self.decoded_map[k] = ele
    
    def predict(self, X):
        Z = X @ self.W
        P = softmax(Z, axis=1)
        encoded_predictions = np.argmax(P, axis=1)
        return np.array([self.decoded_map[x] for x in encoded_predictions])

app = Flask(__name__)

def predict(w):
    csv_data = pd.read_csv("/usr/data/iris.data")
    X = csv_data.iloc[:, :-1].values
    Y = csv_data.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

    logreg = MCLogisticRegressor(X_train, Y_train, w)
    predictions = logreg.predict(X_test)
    return np.sum(predictions == Y_test) / Y_test.shape[0] * 100

@app.route('/server', methods=['POST'])
def server():
    if request.method == 'POST':
        message = request.form.get('message', '')

        with open("/usr/data/temp.txt", 'w') as f:
            f.write(message)
        
        w = None
        with open("/usr/data/temp.txt", "r") as f:
            w = np.loadtxt(f)

        accuracy = predict(w)
        message =  "Accuracy = " + str(accuracy) + "%"
        
        print('Received message from client:', message)
        return 'Message received by the server:' + message
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)