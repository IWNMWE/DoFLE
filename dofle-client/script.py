# Init client script
import requests
import Fed_algo
import os
import time
import logging
import sys

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('Logger')


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


# Base URL of the central server and URLs of all endpoints
baseUrl = "https://nginx"
subscribeUrl = "/subscribe"
sendModelUrl = "/model_updates"
getGlobalModelUrl = "/get_global_model"

# Verify SSL certificate
verify = False

# Time interval(seconds) between successive poll for global model
POLL_INTERVAL = 60

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack= nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

def make_model(model_path):
    model = NeuralNetwork()

    return model


# Load train and test dataset
# def load_dataset():
#     # load dataset
#     (trainX, trainY), (testX, testY) = mnist.load_data()
#     # reshape dataset to have a single channel
#     trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
#     testX = testX.reshape((testX.shape[0], 28, 28, 1))
#     # one hot encode target values
#     trainY = to_categorical(trainY)
#     testY = to_categorical(testY)

#     random_indices = np.random.choice(len(trainX), 10016, replace=False)

#     # Select the subset of data and labels
#     subset_X = trainX[random_indices]
#     subset_Y = trainY[random_indices]
#     return subset_X, subset_Y, testX, testY

# # Scale pixels
# def prep_pixels(train, test):
#     # convert from integers to floats
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     # normalize to range 0-1
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     # return normalized images
#     return train_norm, test_norm


class Client:
    def __init__(self, name=None, modelFile=None, dataFolder=None, lr=0.01) -> None:
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

        # An identifier representing the version of the model with the
        # client with respect to the Federated Learning process
        self.version = None

        # Load dataset and prepare the pixel data
        self.trainX, self.trainY, self.testX, self.testY = load_dataset()
        self.trainX, self.testX = prep_pixels(self.trainX, self.testX)
        self.baseClient = Fed_algo.ClientScaffold(self.trainX, self.trainY, self.testX,
                                                  self.testY, 32, make_model(),
                                                  tf.keras.losses.CategoricalCrossentropy(from_logits=True), [tf.keras.metrics.CategoricalAccuracy()], lr, optim=torch.optim.SGD)

        # Number of datapoints the client model has been trained upon
        self.datapoints = len(self.trainX)

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
            model = tf.keras.models.load_model(modelFile, compile=False)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(
                              from_logits=True),
                          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            return model
        except Exception as exp:
            logger.error("Failed to load model: " + str(exp))

    def loadDataset(self, dataFolder=None):
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
                seed=123,
                image_size=(28, 28),
                batch_size=1)
            return train_ds
        except Exception as exp:
            logger.error("Failed to load dataset: " + str(exp))

    def train(self, train_ds=None, model=None, modelFile=None):
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
            normalized_ds = train_ds.map(
                lambda x, y: (normalization_layer(x), y))
            model.fit(normalized_ds, epochs=1)
            self.datapoints += len(list(normalized_ds))
            model.save(modelFile)
        except Exception as exp:
            logger.error("Failed to train model: " + str(exp))

    def subscribeToFL(self):
        """Calls the subscribe endpoint and subscribes the client to
           a Federated Learning environment. Stores the corresponding
           id in the client for further authorisations.
        """

        url = baseUrl + subscribeUrl
        try:
            response = requests.post(url, json={
                "name": self.name
            }, verify=verify)

            if response.status_code == 200:
                self.id = response.json()["client"]["id"]
                logger.info("Subscribed with id " + str(self.id))
            else:
                logger.error("Response error: " +
                      str(response.status_code) + ":" + response.text)
        except Exception as exp:
            logger.error("Request exception: " + str(exp))

    def pollForGlobalModel(self):
        """Polls the server until it gets the global model. If the client
           is not selected, it polls again. If the client is selected,
           the client can start training the model.
        """

        url = baseUrl + getGlobalModelUrl
        keepPolling = True
        globalModel = None

        def getModel():
            nonlocal globalModel
            try:
                response = requests.get(url, json={
                    "id": self.id
                }, verify=verify)

                if response.status_code == 200:
                    status = response.json()["status"]
                    logger.info(status)
                    if "not" in status:
                        # Poll again after sometime later
                        return True
                    else:
                        # Stop polling
                        globalModel = response.json()["model"]
                        self.version = globalModel["version"]
                        return False
                else:
                    logger.error("Response error: " +
                          str(response.status_code) + ":" + response.text)
            except Exception as exp:
                logger.error("Failed to get FL status: " + str(exp))

            return True

        while keepPolling == True:
            keepPolling = getModel()
            if keepPolling == True:
                # Wait for [POLL_INTERVAL] time
                time.sleep(POLL_INTERVAL)

        # Got global model, start training
        if globalModel is not None:
            C = Fed_algo.convert_tond(globalModel["globalC"])
            Global = Fed_algo.convert_tond(globalModel["weights"])
            delta_weights, delta_C, n = self.baseClient.train(C, Global)
            self.sendModelUpdates(delta_C, delta_weights, n)
            self.pollForGlobalModel()

    def sendModelUpdates(self, delta_C=None, delta_weights=None, datapoints=None):
        """Sends the [model] updates to the server. If the [model] is
           not provided, it uses [modelFile] and client's implementation
           of loading model to get the model.

        Params:    
            modelFile: The modelFile that contains the model to be sent
            model: A model whose updates will be sent
        """

        url = baseUrl + sendModelUrl
        try:
            # if model == None:
            # model = self.loadModel(modelFile = modelFile)

            # weights = model.get_weights().tolist()
            delta_C = Fed_algo.convert_tolist(delta_C)
            delta_weights = Fed_algo.convert_tolist(delta_weights)
            response = requests.post(url, json={
                "id": self.id,
                "version": self.version,
                "delta_weights": delta_weights,
                "delta_C": delta_C,
                "datapoints": datapoints
            }, verify=verify)

            if response.status_code == 204:
                logger.info("Sent model updates")
            elif response.status_code == 200:
                logger.info(response.text)
            else:
                logger.error("Response error: " +
                      str(response.status_code) + ":" + response.text)
        except Exception as exp:
            logger.error("Failed to send model updates: " + str(exp))


# Entry point
if __name__ == '__main__':
    client = Client(modelFile="x.h5", dataFolder="test")
    client.subscribeToFL()
    client.pollForGlobalModel()
