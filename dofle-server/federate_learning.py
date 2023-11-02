# Federated Learning script
from enum import Enum
from __main__ import storage
CLIENT_BATCH_SIZE = 5

class FLMode(Enum):
    WAITING_FOR_SELECTED_CLIENTS = 1
    AGGREGATING = 2
    WAITING_FOR_MODELS = 3 

class FederatedLearningComponent():
    def __init__(self , method , global_lr) -> None:
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

        #The string contaning the method name
        self.method = method

        #global control variate
        if self.method == "scaffold" or self.method == "SCAFFOLD":
            self.global_C = self.global_model[0]["weights"]
            for i in self.global_c:
                i.fill(0)

        #Global learning rate of the server 
        self.global_lr = global_lr

        # Map of client ids that map to model updates
        # sent from selected clients
        self.delta_c = {}
   
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

    def update_global(self, global_weights, delta_weights,
                     nk, delta_c):
        """Performs the global updates to the global model [global_weights] 
           and all the required parameters [C_global] based on the federated 
           learning algorithm used       
        
           Returns : 
                global_model : The weights of the global model.
                global_C     : The global control variate.
                              (optional depending on the algorithm)
        """
        ind = 0

        total_datapoints = 0
        for n in nk:
            total_datapoints = total_datapoints + 1
        
        if(self.method == "scaffold" or self.method == "SCAFFOLD"):

            for delta_weight in delta_weights:
                for i in range(0, len(global_weights)):
                    global_weights[i] = (global_weights[i] 
                                        + (delta_weights[i] 
                                        * (self.global_lr * nk[ind]/float(total_datapoints))))
                    self.global_C[i] = (self.global_C[i] 
                                + (delta_c[i] * (nk[ind]/float(total_datapoints))) 
                                * (len(delta_weight) / float(len(self.clients))))
                
                ind = ind + 1
                
            return [global_weights,global_C]
        
        if(self.method == "avg" or self.method == "AVG"): 
            for delta_weight in delta_weights:
                for i in range(0, len(global_weights)):
                    global_weights[i] = global_weights[i] 
                    + (delta_weights[i] * (self.global_lr * nk[ind]/float(total_datapoints)))
                
                ind = ind + 1

            return [global_weights]

    def server_train(self):
        delta_weights = []
        nk = []
        delta_c = []     
        for _, model in self.client_models:
           model_dict = storage.retrieve(model['model_key'])
           delta_weights.append(model_dict['weights'])
           nk.append(model_dict["datapoints"])
        
        if self.method == "scaffold" or self.method == "SCAFFOLD":
            for _, model in self.client_models:
                model_dict = storage.retrieve(model['model_key'])
                delta_c.append(model_dict['weights'])

        self.update_global(self.global_weights[-1]['weights'], delta_weights,
                     nk, delta_c)
           
