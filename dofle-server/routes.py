# Server routes definitions
from __main__ import app, storage, fed
from federate_learning import FLMode
from flask import request
import numpy as np

currentClientId = 0

CLIENT_NOT_SELECTED = 'Client has not been selected for the process', 200
CLIENT_SELECTED = 'Client has been selected for the process', 200
MODEL_ALREADY_RECEIVED = 'Client\'s model has already been receieved', 200
MODEL_NOT_TRAINED = 'New model is not yet ready', 200
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
            fed.clients.append(client)
            
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
            if not any(x["id"] == id for x in fed.clients):
                return CLIENT_NOT_REGISTERED
            
            if id not in fed.selected_clients:
                msg, _ = CLIENT_NOT_SELECTED
                return msg, 403
            
            if id in fed.client_models.keys():
                return MODEL_ALREADY_RECEIVED
            
            model = {
                "weights": request.json["weights"],
                "datapoints": request.json["datapoints"],
            }
            model_key = storage.store("c", model)
            fed.client_models[id] = {
                "version": request.json["version"],
                "model_key": model_key
            }
            fed.changeMode()

            return NO_CONTENT
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST

@app.route('/fl_process_status', methods=['GET'])
def flProcessStatus():
    """Returns the status of the Federated Learning process by
       informing the client if it has been selected for the process.

    Request Params:    
        id: The id of the client assigned by the FL system

    Returns:
        status: A message signifying the selection of the client
    """

    if request.method == 'GET':
        try:
            id = request.json["id"]
            if not any(x["id"] == id for x in fed.clients):
                return CLIENT_NOT_REGISTERED
            
            msg = ""
            code = None
            if id not in fed.selected_clients:
                msg, code = CLIENT_NOT_SELECTED
            else:
                msg, code = CLIENT_SELECTED
            
            return {"status": msg}, code
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST

@app.route('/get_global_model', methods=['GET'])
def getGlobalModel():
    """Returns the global model if the round of training has
       been completed and the client is selected for the next
       round.

    Request Params:    
        id: The id of the client assigned by the FL system

    Returns:
        status: The status of the client wrt the FL process
        version: The version of the global model
        weights: The weight matrix of the model (as a list)
    """

    if request.method == 'GET':
        try:
            id = request.json["id"]
            
            try:
                if not any(x["id"] == id for x in fed.clients):
                    return CLIENT_NOT_REGISTERED

                if id not in fed.selected_clients:
                    msg, code = CLIENT_NOT_SELECTED
                    return {"status": msg}, code

                if (fed.flMode == FLMode.AGGREGATING or
                    fed.flMode == FLMode.WAITING_FOR_MODELS or
                    fed.global_models):
                    msg, code = MODEL_NOT_TRAINED
                    return {"status": msg}, code

                modelWeights = storage.retrieve(fed.global_models[-1]["model_key"])

                model = fed.global_models[-1].copy()
                model["weights"] = modelWeights

                if id not in fed.selected_clients_with_model:
                    fed.selected_clients_with_model.append(id)
                    fed.changeMode()

                msg, _ = CLIENT_SELECTED
                return {"status": msg, "model": model}, 200
            except:
                return INTERNAL_SERVER_ERROR
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST