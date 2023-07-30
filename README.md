<div align="center">
  <h1 align="center">DoFLE</h1>
</div>
<br/>

## Description

DoFLE is an innovative Docker-based Federated Learning Environment designed for running Federated Averaging algorithms in a distributed network of clients and servers. The platform allows seamless integration and testing of various machine learning models in a federated setting, enabling efficient collaboration and learning without sharing raw data.

## Features

- **Docker Integration:** DoFLE leverages Docker containers to provide a flexible and scalable environment for Federated Learning.

- **Federated Averaging:** The platform implements the Federated Averaging algorithm, enabling model training across multiple clients and servers.

- **Distributed Computing:** DoFLE optimizes performance by exploring distributed computing techniques for faster and more efficient model training.

## Getting Started

1. Clone the DoFLE repository to your local machine.

2. Install Docker and Docker Compose on your system.

3. Open the folder and build a network on docker using the `docker network create my-net` command.

4. Using the server and client folder paths run the command `docker build .`.

5. Now start the docker containers using `docker run --network = my-net --mount type=bind,source="$(pwd)",target=/app  client` and `docker run --network = my-net --mount type=bind,source="$(pwd)",target=/app  server`.(You can start multiple client containers from different ports)

6. Now that the client and server containers are running the client containers will ask for paths of the modelfile and datafolder as paths.(Make sure that the model files are seperate for each client)

after these steps have been done the clients will do training on the mnist data folder present in the project in case model H5 files dont exist withinh your system you can pick fron the readily avalible models folder which can train on the mnist data folder provided.

The containers will only run the training for one round of global model and will save the global model in the model file path provided for a client. To perform multiple rounds the client containers must be started over again and again.

## Results

For the ANN model on division of data in a non IID manner and running for 4 local models and 5 local epochs and 4 rounds gives local accuracy of near 96 percent and global accuarcy of 95 percent

For the CNN model on division of data in a non IID manner and running for 5 local models and 1 local epochs and 10 rounds gives local accuracy of near 99.5 percent and global accuarcy of 98.9 percent
