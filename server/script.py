from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax
import tensorflow as tf
import requests
import asyncio

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


async def sendCurrentModel(clients, model, retry=True, timeout=300, numTimes=1):
    """Sends current model to clients asynchronously.

    Sends a POST request to [clients] to acknowledge they are online
    along with the current model [model] for a current iteration.
    In case client(s) do not respond (offline), they are flagged to be
    requested again after every [timeout] seconds for [numTimes] if
    [retry] == True.

    Args:
        clients: list of chosen clients
        W: current model (weights)
        retry: whether to retry failed requests
        timeout: seconds to wait before retrying
        numTimes: number of times to retry
    """
    for i in range(1 + numTimes):
        retryClients = []            
        for client in clients:
            url = f'http://{client}:8000/client'
            w = model.get_weights()
            for i in range(0,len(w)):
                w[i] =  w[i].tolist()
            response = requests.post(url, json = {"weights" : w,"config" : model.get_config()})

            if response.status_code == 200:
                print(f'Client - {client} received model')
            else:
                if(retry == True):
                    retryClients.append(client)
                print('Error:', response.status_code)
        
        # Break if [retry] = False or ifall requests were successful
        if(retry == False or len(retryClients) == 0):
            break
        # Sleep for [timeout] seconds and retry
        else:
            await asyncio.sleep(timeout)
            clients = retryClients


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

