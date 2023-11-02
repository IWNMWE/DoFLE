# Init server script
from flask import Flask, request
import numpy as np
import asyncio
import tensorflow as tf
import models
import io

# Import storage class
from storage import Storage
storage = Storage()

app = Flask(__name__)

# Import federated learning functionalities
from federate_learning import FederatedLearningComponent
fed = FederatedLearningComponent("scaffold",1)



# Import the server routes
import routes


# Entry point
if __name__ == '__main__':

    global_C = models.load_model()
    weights_stream = io.BytesIO()
    global_C.save_weights(weights_stream)
    
    key = storage.store("w",weights_stream.getvalue().decode('utf-8'))
   
    temp = global_C.get_weights()
    for i in temp:
        i.fill(0)
    global_C.set_weights(temp)
    weights_stream.seek(0)
    weights_stream.truncate()
    global_C.save_weights(weights_stream)
    
    key_dash = storage.store("c",weights_stream.getvalue().decode('utf-8'))

    fed.global_models.append({
        "version" : 1, 
        "model_key" : key,
        "global_C_key"  : key_dash
    })
    fed.selected_clients = fed.selectClientsForRound(fed.clients,fed.selected_clients)

    app.run(host='0.0.0.0', port=8000)
    while True:
        pass