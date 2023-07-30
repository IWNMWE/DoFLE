import tensorflow as tf

"""
This file generates a CNN2D model and saves
the model as an H5 file format

"""

def CNN2D():
 model = Sequential()
 model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 3)))
 model.add(tf.keras.layers.MaxPooling2D((2, 2)))
 model.add(tf.keras.layers.Flatten())
 model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
 model.add(tf.keras.layers.Dense(10, activation='softmax'))
 # compile model
 opt = tf.keras.layers.SGD(learning_rate=0.01, momentum=0.9)
 model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 return model

if __name__ == '__main__':
    model = CNN2D()
    model.save('CNN2D.h5')
