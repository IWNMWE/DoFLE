# Server routes definitions
from __main__ import app, clients
from flask import request

currentClientId = 0

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