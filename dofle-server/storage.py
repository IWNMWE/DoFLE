# Storage manager
import redis
import uuid
import struct
import numpy as np


class Storage:
    def __init__(self) -> None:
        self._map = {}
        self._db = redis.StrictRedis(host="redis", port=6379, db=0)

    def store(self, prefix, model) -> str:
        """Stores the model under a randomly generated string with
            prefix [prefix]. The [model] is packed in a '>II' binary
            format before storing.

        Args:    
            prefix: A string that will be the prefix of the key
            model:  A weight matrix (numpy array) to be stored

        Returns:
            key: The key with which the model is stored.
        """
        key = prefix + str(uuid.uuid4())

        h, w = model.shape
        shape = struct.pack('>II', h, w)
        encoded_model = shape + model.tobytes()

        self._db.set(key, encoded_model)
        return key

    def retrieve(self, key) -> np.ndarray:
        """Retrieves the model stored under key [key]. The model is
            unpacked and converted into an np.ndarray before its
            returned.

        Args:    
            key: A string that is used to identify the model

        Returns:
            model: The stored model interpreted as a np.ndarray.
        """
        encoded = self._db.get(key)
        h, w = struct.unpack('>II', encoded[:8])
        model = np.frombuffer(encoded[8:]).reshape(h, w)
        return model

    def remove(self, key) -> None:
        """Removes the model stored under key [key] from the buffer.

        Args:    
            key: A string that is used to identify the model
        """
        self._db.delete(key)
        return
