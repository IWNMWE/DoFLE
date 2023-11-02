import tensorflow as tf
from tf.keras.models import load_model
import numpy as np

"""
These are all the class implementations of the federated learning
client.Each of these classes have a train method which performs the
local update steps.

Currently supported methods : 
  1.) SCAFFOLD
  2.) FedAvg
"""

class ClientScaffold:
  def __init__(self , trainX , trainY , testX , 
                testY , batchSize , model_path , 
                loss , metrics , lr, optim = tf.keras.optimizers.SGD()):

    self.model = load_model(model_path)
    self.c = self.initializeC()
    self.cPlus = self.initializeC()
    self.trainX = trainX
    self.trainY = trainY
    self.testX = testX
    self.testY = testY
    self.batch = batchSize
    self.lr = float(lr)
    self.losses = loss
    self.metrics = metrics
    self.optim  = optim
    self.optim.learning_rate = self.lr
  def train(self , C , Global):
    self.model.compile(optimizer = self.optim , loss = self.losses , metrics = self.metrics)
    self.model.set_weights(Global)
    num_batches = len(self.trainX) // self.batch

    # Split the data into batches
    batches = np.array_split(self.trainX, self.batch)
    batchesY = np.array_split(self.trainY,self.batch)
    remaining_data = self.trainX[num_batches * self.batch:]
    if len(remaining_data) > 0:
        batches.append(remaining_data)
        batchesY.append(self.trainY[num_batches * self.batch:])

    for i in range(0,len(batches)):
      self.model.fit(
      batches[i],batchesY[i],verbose  = 0
      )
      weights = list(self.model.get_weights())
      for i in range (0 , len(Global)):
        weights[i] += (1 / (len(batches) * self.lr)) * (C[i] - self.c[i])
      self.model.set_weights(weights)

    weights = list(self.model.get_weights())
    delta_weights = list(weights)
    delta_C = list(self.cPlus)
    for i in range (0 , len(weights)):
        self.cPlus[i] = self.c[i] - C[i] + (1 / ((1 / (len(batches) * self.lr))  
                        * (len(self.trainX) / self.batch))) * (Global[i] - weights[i])
        delta_weights[i] -=  Global[i]
        delta_C[i] -= self.c[i]
    self.c = list(self.cPlus)
    
    return [delta_weights,delta_C,len(self.trainX)]
  
  def initializeC(self):
    C = self.model.get_weights()
    for i in C:
      i.fill(0)
    return C

class ClientFedAvg:
  def __init__(self , trainX , trainY , testX , 
                testY , batchSize , model_path , 
                 loss , metrics , lr , optim  = SGD()):

    self.model = load_model(model_path)
    self.trainX = trainX
    self.trainY = trainY
    self.testX = testX
    self.testY = testY
    self.batch = batchSize
    self.lr = optim.learning_rate
    self.losses = loss
    self.metrics = metrics
    self.optim  = optim
  def train(self , C , Global , epoch = 5):
    self.model.compile(optimizer = self.optim , loss = self.loss , metrics = self.metrics)
    self.model.set_weights(Global)

    self.model.fit(
    self.trainX , self.trainY , validation_data=(self.testX , 
    self.testY) , epochs = epoch)
    weights = list(self.model.get_weights())
    delta_weights = list(weights)
    for i in range (0 , len(weights)):
        delta_weights[i] -=  Global[i]
    
    return [delta_weights,len(self.trainX)]
