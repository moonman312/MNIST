import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from MNIST_Classifier import MNIST_Classifier as Model
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
from PIL import Image
import numpy as np

#Set up env so that nothing is sent to stdout unless there is a failure
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.FATAL)

def train():
    print("Loading IDX Training Data")
    input_data.read_data_sets('./data')
    with open('./data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = extract_images(f)
    with open('./data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = extract_labels(f)
    print("Loading IDX Test Data")
    with open('./data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = extract_images(f)
    with open('./data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = extract_labels(f)
    print("Preprocessing Data")
    #normalize between 0 and 1
    train_images = train_images / 255
    test_images = test_images / 255
    model = Model()
    print("Training Model")
    model.train_model(train_images, train_labels, test_images, test_labels)
    print("Evaluating model")
    loss, accuracy = model.eval(test_images, test_labels)
    print(f'Evaluation - Loss: {loss} Acc: {accuracy}')
    model.save_model('./MNIST_Classifier.h5')

def predict(model, image_file_path):
    file = tf.read_file(image_file_path)
    image = tf.image.decode_png(file, channels=1)
    image = tf.image.resize_images(image, size=(28, 28))
    image = tf.cast(image, dtype=tf.uint8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image = tf.expand_dims(image, 0)
        image = image.eval()
        #normalize between 0 and 1
        image = image / 255
        print(f'Predicting {image_file_path}')
        prediction = model.predict_image(image)
        print(f'Prediction: \'{prediction[0][0]}\' Confidence: {prediction[1]}%')
################################################################################
################################################################################
########################Logic For Parsing Arguments#############################
################################################################################
################################################################################
parser = argparse.ArgumentParser(description='Determine action')
parser.add_argument('--predict', help='Predict value of image')
parser.add_argument('--train', help='Train on stored data')
parser.add_argument('image_file_path', nargs='*', help='file path of image you would like to predict', default='./Handwritten-digit-2.png')

args = parser.parse_args()
if args.predict:
    model = Model()
    model.set_model(model.load_model_and_weights('./model_architecture.json', './model_weights.h5'))
    predict(model, args.image_file_path)
else:
    train()
