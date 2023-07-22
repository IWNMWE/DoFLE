from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax
import tensorflow as tf
import docker

app = Flask(__name__)



def FSGD(weights,config):
    final_weights = []
    for i in weights[0]["weights"]:
        final_weights.append(np.zeros(shape = i.shape))
    total = 0.0
    for i in weights:
      total += i["datapoints"]
    for i in range (0,len(weights)):
        for j in range (0,len(weights[i]["weights"])):
          final_weights[j] = np.add(final_weights[j], (weights[i]["datapoints"] / total) * weights[i]["weights"][j])
    new_model = keras.Sequential.from_config(config) 
    new_model.set_weights(final_weights)
    return new_model 




def getClientContainers():
    return docker.from_env().containers.list(filters={"name": "client"})

@app.route('/server', methods=['POST'])
def recieve():
        
    if request.method == 'POST':
        config = request.json["config"]
        model = tf.keras.Sequential().from_config(config)
        weights = request.json["weights"]
        datapoints = request.json["datapoints"]
        id = request.json["id"]
        for i in range(0,len(weights)):
            weights[i] =  np.array(weights[i])
        models[id] = {"weights" : weights,"datapoints":datapoints}
        return "Got model from your machine"
        
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

