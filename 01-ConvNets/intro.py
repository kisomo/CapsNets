import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as p

#from data_prep import prepare_data as prd
#import data_prep 

#http://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-tensorflow/

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


#load the MNIST and  CIFAR-10 datasets 
import mnist
from mnist import MNIST
mndata = MNIST('/home/terrence/CODING/Python/MODELS/CapsNets/python-mnist/data')

mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
 
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
 
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_width, mnist_image_height, mnist_image_depth)
 
print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_width*mnist_image_height*1))
print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))
 
print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
 
train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
 
######################################################################################
print("+++ cifar10 ++++++++++")

cifar10_folder = '/home/terrence/CODING/Python/MODELS/CapsNets/data/cifar10/'
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
 

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    #c10_test_dict = p.load(f0, encoding='bytes')
    c10_test_dict = p.load(f0)
c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_height, c10_image_width, c10_image_depth)
 

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        #c10_train_dict = p.load(f0, encoding='bytes')
        c10_train_dict = p.load(f0)
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
 
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_
 
c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_height, c10_image_width, c10_image_depth)
del c10_train_dataset
del c10_train_labels
 
print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print('Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)



# fit a 1 layer FCNN

image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels 
batch_size = 10 # I just created this

#the dataset
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels 
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels 


'''
train_dataset = 10
train_labels = 10
test_dataset = 10
test_labels = 10

#mn = prepare_data("mnist")
mnist = data_prep.prepare_data("mnist",train_dataset,train_labels,test_dataset,test_labels)
mnist = prepare_data._display(mnist)
#number of iterations and learning rate
'''
num_steps = 1001  #10001
display_step = 100  #1000
learning_rate = 0.5

 
graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
  
    #2) Then, the weight matrices and bias vectors are initialized
    #as a default, tf.truncated_normal() is used for the weight matrix and tf.zeros() is used for the bias vector.
    weights = tf.Variable(tf.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32)
    bias = tf.Variable(tf.zeros([num_labels]), tf.float32)
  
    #3) define the model:
    #A one layered fccd simply consists of a matrix multiplication
    def model(data, weights, bias):
        return tf.matmul(flatten_tf_array(data), weights) + bias
        #return tf.nn.xw_plus_b(train_dataset, weights, bias)  # Terrence exchanged this
 
    logits = model(tf_train_dataset, weights, bias)
    #4) calculate the loss, which will be used in the optimization of the weights
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
 
    #5) Choose an optimizer. Many are available.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    #6) The predicted values for the images in the train dataset and test dataset are assigned to the variables train_prediction and test_prediction. 
    #It is only necessary if you want to know the accuracy by comparing it with the actual values. 
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, bias))
 

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % display_step == 0):
            train_accuracy = accuracy(predictions, train_labels[:, :])
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)













