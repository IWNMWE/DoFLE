# Server routes definitions
from __main__ import app, storage, fed
from federate_learning import FLMode
from flask import request

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
        client: A client map containing information to uniquely identify
                a client in further process
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
            fed.shouldSelectClient()

            return {
                "client": client,
            }, 200
        except:
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST


@app.route('/model_updates', methods=['POST'])
def receiveModelUpdates():
    """Receives model updates sent by a selected client.

    Request Params:    
        id: The id of the client assigned by the FL system
        delta_weights: The serialised weight difference matrix from the trained model
        delta_C: The serialised control variate difference matrix from the trained model
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
                "delta_weights": request.json["delta_weights"],
                "delta_C": request.json["delta_C"],
                "datapoints": request.json["datapoints"],
            }
            model_key = storage.store("c", model)
            fed.client_models[id] = {
                "version": request.json["version"],
                "model_key": model_key
            }
            fed.changeMode()

            return NO_CONTENT
        except Exception as e:
            print(str(e))
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST

# !!!Deprecated : Poll for the global model directly


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
        weights: The serialised weight matrix of the global model
        globalC: The serialised control variate matrix of the global model
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
                    id in fed.client_models.keys() or
                        len(fed.global_models) == 0):
                    msg, code = MODEL_NOT_TRAINED
                    return {"status": msg}, code

                modelWeights = storage.retrieve(
                    fed.global_models[-1]["model_key"])
                globalC = storage.retrieve(
                    fed.global_models[-1]["global_C_key"])

                model = {
                    "version": fed.global_models[-1]["version"],
                    "weights": modelWeights,
                    "globalC": globalC
                }

                if id not in fed.selected_clients_with_model:
                    fed.selected_clients_with_model.append(id)
                    fed.changeMode()

                msg, _ = CLIENT_SELECTED
                return {"status": msg, "model": model}, 200
            except Exception as e:
                print(str(e))
                return INTERNAL_SERVER_ERROR
        except Exception as e:
            print(str(e))
            return UNPROCESSABLE_ENTITY
    else:
        return INVALID_REQUEST
