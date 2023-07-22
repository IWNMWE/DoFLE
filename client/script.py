import requests
import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy .special import softmax
import tensorflow as tf
from flask import Flask, request

hostname = os.environ['HOSTNAME']

modelfile = ' '
datafolder = ' '

datapoints = 0
# Client class implementation for clients

class client:

  # Modelfile is the model stored in the H5 format datafolder should 
  # be an image classification data set which can be passed in the 
  # tensorflow image dataset from directory method 
  def __init__(self,modelfile,datafolder):
    self.modelfile = modelfile
    self.datafolder = datafolder
  
  #Load model method
  def loadmodel(self,modelfile = None):
    if modelfile is None:
      modelfile = self.modelfile
    model = tf.keras.models.load_model(modelfile,compile = False)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model
  
  #Loading the image classification data set
  def load_dataset(self,datafolder = None):
    if datafolder is None:
      datafolder = self.datafolder

    train_ds = tf.keras.utils.image_dataset_from_directory(
              datafolder,
              seed = 123,
              image_size=(28, 28),
              batch_size = 1)
    return train_ds

  # Prepares a normalized dataset and trains the model
  # stores the model in the modelfile directory
  def train(self,train_ds = None,model = None,modelfile = None):
    
    global datapoints
    if train_ds is None:
      train_ds = self.load_dataset()
    if model is None:
      model = self.loadmodel()
    if modelfile is None:
      modelfile = self.modelfile
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x,y : (normalization_layer(x),y))
    datapoints = datapoints + len(list(normalized_ds))
    model.fit(normalized_ds,epochs = 1)
    model.save(modelfile)

  def send_server(self):
    global datapoints
    url = 'http://server:8000/server'  # Replace with the server URL
    model = self.loadmodel()
    w = model.get_weights()
    for i in range(0,len(w)):
          w[i] =  w[i].tolist()
    response = requests.post(url, json = {"weights" : w,"config" : model.get_config(),"datapoints":datapoints})

    if response.status_code == 200:
        print('Server response:', response.text)
    else:
        print('Error:', response.status_code)
  
## Flask app is run to recive the model
app = Flask(__name__)

@app.route('/client', methods=['POST'])
def recieve():
  if request.method == 'POST':
    config = request.json["config"]
    model = tf.keras.Sequential().from_config(config)
    weights = request.json["weights"]
    for i in range(0,len(weights)):
          weights[i] =  np.array(weights[i])
        
    model.set_weights(weights)
    model.save(modelfile)

  else:
        return 'Invalid request method'

   

if __name__ == '__main__':
    modelfile = input() 
    datafolder = input() 
    cli = client(modelfile,datafolder)
    cli.train()
    cli.send_server()
    app.run(host = '0.0.0.0',port = 5000)

