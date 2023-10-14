# Server routes definitions
from __main__ import app, clients, selected_clients, client_models, storage
from flask import request
import numpy as np

currentClientId = 0

CLIENT_NOT_SELECTED = 'Client has not been selected for the process', 200
MODEL_ALREADY_RECEIVED = 'Client\'s model has already been receieved', 200
NO_CONTENT = '', 204
CLIENT_NOT_REGISTERED = 'No matching client id found', 403
INVALID_REQUEST = 'Invalid request method', 405
UNPROCESSABLE_ENTITY = 'Unprocessable entity', 422
INTERNAL_SERVER_ERROR = 'Internal server error', 500


@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Stores the client information allowing them to subscribe
       to the federated learning system.

    Request Params:    
        name: A non-unique string identifer for the client

    Returns:
        id: A unique integer to identify client in further processes 
    """

    if request.method == 'POST':
        global currentClientId

        try:
            name = request.json["name"]
            id = currentClientId
            currentClientId += 1
            client = {
                "id": id,
                "name": name
            }
            clients.append(client)
            
            return {
                "client": client,
            }, 200
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST

@app.route('/model_updates', methods=['POST'])
def receiveModelUpdates():
    """Receives models sent by a selected client.

    Request Params:    
        id: The id of the client assigned by the FL system
        weights: The weight matrix of the model (as a list)
        datapoints: The number of datapoints the client has trained on
        version: The version of the model the client has trained on
    """
        
    if request.method == 'POST':
        try:
            id = request.json["id"]
            if not any(x["id"] == id for x in clients):
                return CLIENT_NOT_REGISTERED
            
            if id not in selected_clients:
                msg, _ = CLIENT_NOT_SELECTED
                return msg, 403
            
            if id in client_models.keys():
                return MODEL_ALREADY_RECEIVED
            
            model = {
                "weights": np.asarray(request.json["weights"]),
                "datapoints": request.json["datapoints"],
            }
            model_key = storage.store("c", model)
            client_models[id] = {
                "version": request.json["version"],
                "model_key": model_key
            }

            return NO_CONTENT
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST