import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


"""
These are all the class implementations of the federated learning
client.Each of these classes have a train method which performs the
local update steps.

Currently supported methods : 
  1.) SCAFFOLD
  2.) FedAvg
  3.) FedProx
"""

class ClientScaffold:
    def __init__(self, dataloader, batchSize, model,
                 loss, metrics, lr, optim):

        self.model = model
        self.c = self.initializeC()
        self.cPlus = self.initializeC()
        self.dataloader = dataloader
        self.batch = batchSize
        self.lr = float(lr)
        self.loss_fn = loss
        self.metrics = metrics
        self.optimizer = optim
        self.optimizer.learning_rate = self.lr

    def train_loop(self, C, Global):
      size = len(self.dataloader.dataset)
      # Set the model to training mode - important for batch normalization and dropout layers
      self.model.train()
      for batch, (X, y) in enumerate(self.dataloader):
        # Compute prediction and loss
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        i = 0
        with torch.no_grad():
          for param in self.model.parameters():
              param.data = param.data + ((1 / (len(self.batch) * self.lr)) * \
                          (C[i] - self.c[i]))
              i += 1
      weights = []
      with torch.no_grad():
          for param in self.model.parameters():
              weights.append(param.data)

      delta_weights = list(weights)
      delta_C = list(self.cPlus)#
      for i in range(0, len(weights)):
            self.cPlus[i] = self.c[i] - C[i] + (1 / ((1 / (len(self.batch) * self.lr))
                * (len(self.trainX) / self.batch))) * (Global[i] - weights[i])
            delta_weights[i] -= Global[i]
            delta_C[i] = self.cPlus[i] - self.c[i]
      self.c = list(self.cPlus)
      return delta_weights, delta_C, size

    def initializeC(self):
        C = []
        with torch.no_grad():
          for param in self.model.parameters():
              C.append(param.data)
        for i in C:
            i.fill_(0)
        return C
