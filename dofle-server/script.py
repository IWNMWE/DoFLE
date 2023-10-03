# Init server script
from flask import Flask, request
import numpy as np
import asyncio
import tensorflow as tf

app = Flask(__name__)

# Import the server routes
import routes

# Entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    while True:
        pass