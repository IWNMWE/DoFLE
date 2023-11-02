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
        """Stores model under a randomly generated string with
            prefix [prefix].

        Args:    
            prefix: A string that will be the prefix of the key
            model:  A json with fields containing a weights matrix
                    (json-serialisable data type) and the datapoints
                    to be stored

        Returns:
            key: The key with which the model is stored.
        """
        key = prefix + str(uuid.uuid4())

        json_model = model.copy()
        json_dumps = json.dumps(json_model)

        self._db.set(key, json_dumps)
        return key

    def retrieve(self, key):
        """Retrieves the model stored under key [key].

        Args:    
            key: A string that is used to identify the model

        Returns:
            model: A json with fields containing a weights matrix
                   and datapoints
        """
        model = json.loads(self._db.get(key))

        return model

    def remove(self, key) -> None:
        """Removes the model stored under key [key] from the buffer.

        Args:    
            key: A string that is used to identify the model
        """
        self._db.delete(key)
        return
