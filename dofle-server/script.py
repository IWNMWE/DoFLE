# Init server script
from flask import Flask, request
import numpy as np
import asyncio
import tensorflow as tf
import models

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

    global_C = models.load_model().get_weights()
    key = storage.store("w",global_C)
    for i in global_C:
        i.fill(0)
    key_dash = storage.store("c",global_C)
    fed.global_models.append({
        "version" : 1, 
        "model_key" : key,
        "global_C_key"  : key_dash
    })
    fed.selected_clients = fed.selectClientsForRound(fed.clients,fed.selected_clients)

    app.run(host='0.0.0.0', port=8000)
    while True:
        pass