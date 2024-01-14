# Init server script
from flask import Flask
import tensorflow as tf
import models
from waitress import serve

# Import storage class
from storage import Storage
storage = Storage()

app = Flask(__name__)

# Import federated learning functionalities
from federate_learning import FederatedLearningComponent
fed = FederatedLearningComponent("scaffold",1, 3)



# Import the server routes
import routes

# Deployment mode
mode = "Prod"

# Entry point
if __name__ == '__main__':

    # Initialise the xfirst global model
    global_C = models.load_model()

    weights = models.arrayToList(global_C.get_weights())
    key = storage.store("w", weights)

    temp = global_C.get_weights()
    for i in temp:
        i.fill(0)

    temp = models.arrayToList(temp)
    key_dash = storage.store("g", temp)

    fed.global_models.append({
        "version" : 1, 
        "model_key" : key,
        "global_C_key"  : key_dash
    })

    # Start the flask app
    if mode == "Dev":
        app.run(host='0.0.0.0', port=8000)
    else:
        serve(app, host='0.0.0.0', port=8000)
    while True:
        pass