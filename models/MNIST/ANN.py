import tensorflow as tf

"""
This file generates an ANN model and saves
the model as an H5 file format

"""

def ANN():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

if __name__ == '__main__':
    model = ANN()
    model.save('.\ANN.h5')