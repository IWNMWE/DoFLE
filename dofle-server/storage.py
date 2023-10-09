# Storage manager
import redis
import uuid
import numpy as np
import json


class Storage:
    def __init__(self) -> None:
        self._map = {}
        self._db = redis.StrictRedis(host="redis", port=6379, db=0)

    def store(self, prefix, model) -> str:
        """Stores the model under a randomly generated string with
            prefix [prefix]. The weights matrix is converted to a list
            before storing.

        Args:    
            prefix: A string that will be the prefix of the key
            model:  A json with fields containing a weights matrix (numpy array)
                    and the datapoints to be stored

        Returns:
            key: The key with which the model is stored.
        """
        key = prefix + str(uuid.uuid4())

        json_model = model.copy()
        json_model["weights"] = model["weights"].tolist()
        json_dumps = json.dumps(json_model)

        self._db.set(key, json_dumps)
        return key

    def retrieve(self, key) -> np.ndarray:
        """Retrieves the model stored under key [key]. The weights 
            matrix is converted into an np.ndarray before
            it's returned.

        Args:    
            key: A string that is used to identify the model

        Returns:
            model: A json with fields containing a weights matrix
            (interpreted as a np.ndarray) and datapoints
        """
        model = json.loads(self._db.get(key))

        model["weights"] = np.asarray(model["weights"])
        return model

    def remove(self, key) -> None:
        """Removes the model stored under key [key] from the buffer.

        Args:    
            key: A string that is used to identify the model
        """
        self._db.delete(key)
        return
