# Init client script
import requests
import numpy as np
import tensorflow as tf
import os
import time

# Base URL of the central server and URLs of all endpoints
baseUrl = "http://server:8000"
subscribeUrl = "/subscribe"
sendModelUrl = "/model_updates"
getGlobalModelUrl = "/get_global_model"


POLL_INTERVAL = 60

class Client:
    def __init__(self, name=None, modelFile=None, dataFolder=None) -> None:
        # A string identifer for this particular client, defaults to
        # the Hostname
        self.name = name
        if self.name == None:
            self.name = os.environ['HOSTNAME']

        # An integer identifer for this particular client assigned
        # by a Federated Learning environment
        self.id = None

        # A h5 file containing the trained model
        self.modelFile = modelFile

        # A directory containing the training data set
        self.dataFolder = dataFolder

        # Number of datapoints the client model has been trained upon
        self.datapoints = 0

        # An identifier representing the version of the model with the
        # client with respect to the Federated Learning process
        self.version = None

    def loadModel(self, modelFile=None):
        """Loads a model from the [modelFile]. If it is not provided,
           the client's default modelFile is used. Atleast one of them
           should be non-None.

        Params:    
            modelFile: Path of the h5 file for the model to be loaded

        Returns:
            model: The loaded and compiled model
        """
        
        if modelFile == None:
            modelFile = self.modelFile
        
        try:
            model = tf.keras.models.load_model(modelFile,compile = False)
            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            return model
        except Exception as exp:
            print("Failed to load model: " + str(exp))
    
    def loadDataset(self,dataFolder = None):
        """Loads the dataset from the [dataFolder] directory. If the
           path is not provided, it will default to client's dataFolder.
           Atleast one of them should be non-None.

        Params:    
            dataFolder: Path of the directory from where dataset is to be
                        loaded

        Returns:
            train_ds: The loaded training data set
        """

        if dataFolder == None:
            dataFolder = self.dataFolder

        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataFolder,
                    seed = 123,
                    image_size=(28, 28),
                    batch_size = 1)
            return train_ds
        except Exception as exp:
            print("Failed to load dataset: " + str(exp))
    
    def train(self,train_ds = None,model = None,modelFile = None):
        """Trains the [model] with the [train_ds] and saves it as a h5
           model into the [modelFile] file. In case any of these params
           are None, they are loaded through the client's interface.

        Params:    
            train_ds: A dataset for the model to be trained upon
            model: A model that has to be trained
            modelFile: The file to which the trained model has to be
                       saved to
        """

        if train_ds == None:
            train_ds = self.loadDataset()
        if model == None:
            model = self.loadModel()
        if modelFile == None:
            modelFile = self.modelFile
        
        try:
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            normalized_ds = train_ds.map(lambda x,y : (normalization_layer(x),y))
            model.fit(normalized_ds,epochs = 1)
            self.datapoints += len(list(normalized_ds))
            model.save(modelFile)
        except Exception as exp:
            print("Failed to train model: " + str(exp))

    def subscribeToFL(self):
        """Calls the subscribe endpoint and subscribes the client to
           a Federated Learning environment. Stores the corresponding
           id in the client for further authorisations.
        """

        url = baseUrl + subscribeUrl
        try:
            response = requests.post(url, json={
                "name": self.name
            })

            if response.status_code == 200:
                self.id = response.json()["client"]["id"]
                print("Subscribed with id " + str(self.id))
            else:
                print("Response error: " +
                      str(response.status_code) + ":" + response.text)
        except Exception as exp:
            print("Request exception: " + str(exp))

    def pollForGlobalModel(self):
        """Polls the server until it gets the global model. If the client
           is not selected, it polls again. If the client is selected,
           the client can start training the model.
        """

        url = baseUrl + getGlobalModelUrl
        keepPolling = True
        globalModel = None

        def getModel():
            try:
                response = requests.get(url, json={
                    "id": self.id
                })

                if response.status_code == 200:
                    status = response.json()["status"]
                    print(status)
                    if "not" in status:
                        # Poll again after sometime later
                        return True
                    else:
                        # Stop polling
                        globalModel = response.json()["model"]
                        return False
                else:
                    print("Response error: " +
                        str(response.status_code) + ":" + response.text)
            except Exception as exp:
                print("Failed to get FL status: " + str(exp))
            
            return True

        while keepPolling == True:
            keepPolling = getModel()
            if keepPolling == True:
                # Wait for [POLL_INTERVAL] time
                time.sleep(POLL_INTERVAL)

        # Got global model, start training
        if globalModel is not None:
            pass
        
    def sendModelUpdates(self, modelFile = None,model = None):
        """Sends the [model] updates to the server. If the [model] is
           not provided, it uses [modelFile] and client's implementation
           of loading model to get the model.

        Params:    
            modelFile: The modelFile that contains the model to be sent
            model: A model whose updates will be sent
        """

        url = baseUrl + sendModelUrl
        try:
            if model == None:
                model = self.loadModel(modelFile = modelFile)

            weights = model.get_weights().tolist()
            response = requests.post(url, json={
                "id": self.id,
                "version": self.version,
                "weights": weights,
                "datapoints": self.datapoints
            })

            if response.status_code == 204:
                print("Sent model updates")
            elif response.status_code == 200:
                print(response.text)
            else:
                print("Response error: " +
                      str(response.status_code) + ":" + response.text)
        except Exception as exp:
            print("Failed to send model updates: " + str(exp))

# Entry point
if __name__ == '__main__':
    # client = Client(modelFile="x.h5", dataFolder="test")
    # client.train()
    # client.subscribeToFL()
    # client.pollForStatus()
    while True:
        pass
