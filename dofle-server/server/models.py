# Model operations helper file

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

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


# Load the model
def load_model():
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
#     return trainX, trainY, testX, testY

# Scale pixels
# def prep_pixels(train, test):
#     # convert from integers to floats
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     # normalize to range 0-1
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     # return normalized images
#     return train_norm, test_norm

# # Serialise ndarray to a list
# def arrayToList(array: np.ndarray):
#     new = array.copy()
#     for i in range(0, len(new)):
#         new[i] = new[i].tolist()

#     return new

# # Deserialise list into a ndarray
# def listToArray(l: list):
#     new = l.copy()
#     for i in range(0, len(new)):
#         new[i] = np.asarray(new[i],  dtype='float32')

#     return new
