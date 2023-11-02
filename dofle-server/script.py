# Init server script
from flask import Flask, request
import numpy as np
import asyncio
import tensorflow as tf

# Import storage class
from storage import Storage
storage = Storage()

app = Flask(__name__)

# Import federated learning functionalities
from federate_learning import FederatedLearningComponent
fed = FederatedLearningComponent("scaffold")



# Import the server routes
import routes


# Entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    while True:
        pass