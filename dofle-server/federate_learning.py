# Federated Learning script
from enum import Enum

CLIENT_BATCH_SIZE = 5

class FLMode(Enum):
    WAITING_FOR_SELECTED_CLIENTS = 1
    AGGREGATING = 2
    WAITING_FOR_MODELS = 3 

class FederatedLearningComponent():
    def __init__(self) -> None:
        # List of subsribed client ids
        self.clients = []

        # List of client ids selected for the current
        # round of Federated Averaging
        self.selected_clients = []

        # Map of client ids that map to model updates
        # sent from selected clients
        self.client_models = {}

        # List of version numbers that map to global
        # models
        self.global_models = []

        # Subset of selected clients who have receieved
        # the current global model
        self.selected_clients_with_model = []

        # Mode the federated learning system is in
        self.flMode = FLMode.WAITING_FOR_SELECTED_CLIENTS

    def selectClientsForRound(clients, prevSelection=None):
        """Selects [CLIENT_BATCH_SIZE] number of clients from the [clients]
        list in a round robin manner by starting from the element next of
        the last element of [prevSelection].

        Params:    
            clients: A list of clients to choose the batch from
            prevSelection: The list of previously chosen clients, same type
                        as that of [clients] list

        Returns:
            selected_clients: A list of clients selected for the current batch
        """

        selected_clients = []

        # There will be no effect of a client selection algorithm in this case
        if len(clients) <= CLIENT_BATCH_SIZE:
            print("WARNING : Insufficient clients for batch")
            for client in clients:
                selected_clients.append(client)
            return selected_clients
        
        # Get the index of the last client in the previous selection (if any)
        lastClientIndex = -1
        if len(prevSelection) > 0:
            lastClientIndex = [i for i in range(len(clients))
                if clients[i] == prevSelection[-1]][0]
        
        # Select the clients starting from the next element, in cyclic manner
        i = lastClientIndex + 1
        for _ in range(CLIENT_BATCH_SIZE):
            if i >= len(clients):
                i = 0
            selected_clients.append(clients[i])
            i += 1
        
        return selected_clients

    def clientsToIds(self):
        """Converts the client list into a list of corresponding ids.
        
        Returns:
            ids: A list of client ids
        """
        
        return [x["id"] for x in self.clients]
    
    def changeMode(self):
        """Changes the mode if the necessary conditions have been met,
           and performs the necessary actions upon mode change.
        """

        if self.flMode == FLMode.WAITING_FOR_MODELS:
            if (len(self.selected_clients) ==
                len(self.client_models)):
                self.flMode = FLMode.AGGREGATING
        
        if self.flMode == FLMode.WAITING_FOR_SELECTED_CLIENTS:
            if (len(self.selected_clients) ==
                len(self.selected_clients_with_model)):
                self.selected_clients_with_model = []
                self.selected_clients = self.selectClientsForRound(
                    self.clientsToIds(),
                    prevSelection=self.selected_clients
                )
                self.flMode = FLMode.WAITING_FOR_MODELS
