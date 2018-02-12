import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as p

class prepare_data1:
    #methods for one-hot encoding the labels, loading the data in a randomized array and a method for
    # flattening an array (since a fully connected network needs an flat array as its input):
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
    
    def one_hot_encode(np_array):
        return (np.arange(10) == np_array[:,None]).astype(np.float32)
    
    def reformat_data(dataset, labels, image_width, image_height, image_depth):
        np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
        np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
        np_dataset, np_labels = randomize(np_dataset_, np_labels_)
        return np_dataset, np_labels
    
    def flatten_tf_array(array):
        shape = array.get_shape().as_list()
        return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
    
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])