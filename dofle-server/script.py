# Init server script
from flask import Flask, request
import numpy as np
import asyncio
import tensorflow as tf

app = Flask(__name__)

# List of subsribed client ids
clients = []

# List of client ids selected for the current
# round of Federated Averaging
selected_clients = []

# List of client ids that map to model updates
# sent from selected clients
client_models = []

# List of version numbers that map to global
# models
global_models = []

# Import the server routes
import routes

# Import storage class
from storage import Storage
storage = Storage()

# Entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    while True:
        pass