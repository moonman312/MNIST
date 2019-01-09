import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from EpochLogger import EpochLogger
import h5py
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import os

class MNIST_Classifier():
    def __init__(self):
        self.model = None

    #takes an array of training images and training labelsself
    #fits and returns a Sequential neural network with 4 layers
    def train_model(self, train_images, train_labels, test_images, test_labels):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28,28,1)),
          tf.keras.layers.Dense(784, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        logger = ModelCheckpoint('./')
        num_epochs = 5
        model.fit(train_images, train_labels, epochs=num_epochs, verbose=0, callbacks=[EpochLogger((test_images, test_labels), num_epochs)])
        self.model = model
        return self.model

    #returns a value prediction and a probability
    def predict_image(self, image):
        num = (self.model.predict_classes(image))
        probs = self.model.predict(image)[0]
        probs = probs.tolist()
        return num, max(probs)

    def load_model_and_weights(self, model_architecture_file_path, model_weights_file_path):
        # load json and create model
        json_file = open(model_architecture_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, {'GlorotUniform': glorot_uniform()})
        # load weights into new model
        loaded_model.load_weights(model_weights_file_path)
        loaded_model.save('model.hdf5')
        loaded_model = load_model('model.hdf5')
        self.model = loaded_model
        return self.model

    def save_model(self, file_path):
        model_json = self.model.to_json()
        with open('model_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('model_weights.h5')
        del self.model
        return

    def eval(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels, verbose=0)

    def set_model(self, model):
        self.model = model
        return
