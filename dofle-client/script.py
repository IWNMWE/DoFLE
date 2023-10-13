# Init client script
import requests
import numpy as np
import tensorflow as tf
import os


# Base URL of the central server and URLs of all endpoints
baseUrl = "http://server:8000"
subscribeUrl = "/subscribe"

class Client:
    def __init__(self, name=None) -> None:
        # A string identifer for this particular client, defaults to
        # the Hostname
        self.name = name
        if self.name == None:
            self.name = os.environ['HOSTNAME']

        # An integer identifer for this particular client assigned
        # by a Federated Learning environment
        self.id = None

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
                      str(response.status_code) + ":" + response.json)
        except Exception as exp:
            print("Request exception: " + str(exp))


# Entry point
if __name__ == '__main__':
    client = Client()
    client.subscribeToFL()
    while True:
        pass
