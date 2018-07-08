import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as p

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
 
print('MNIST Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('MNIST Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
 
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
 
print("CIFAR The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print('CIFAR Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('CIFAR Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)



# +++++++++++++++++++fit a 1 layer FCNN to mnist dataset  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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


#number of iterations and learning rate
num_steps = 101  #10001
display_step = 10  #1000
learning_rate = 5

 
graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
  
    #2) Then, the weight matrices and bias vectors are initialized
    #as a default, tf.truncated_normal() is used for the weight matrix and tf.zeros() is used for the bias vector.
    weights = tf.Variable(tf.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32)
    bias = tf.Variable(tf.ones([num_labels]), tf.float32)
  
    #3) define the model:
    #A one layered fccd simply consists of a matrix multiplication
    def model(data, weights, bias):
        return tf.matmul(flatten_tf_array(data), weights) + bias
        #return tf.nn.xw_plus_b(train_dataset, weights, bias)  # Terrence exchanged this
 
    logits = model(tf_train_dataset, weights, bias)
    #4) calculate the loss, which will be used in the optimization of the weights
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    print("Terrence", loss)
    #5) Choose an optimizer. Many are available.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    #6) The predicted values for the images in the train dataset and test dataset are assigned to the variables train_prediction and test_prediction. 
    #It is only necessary if you want to know the accuracy by comparing it with the actual values. 
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, bias))
 

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('mnist training Initialized')
    for step in range(num_steps):
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #_, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % display_step == 0):
            train_accuracy = accuracy(predictions, train_labels[:, :])
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)


# +++++++++++++++++++fit a 1 layer FCNN to CIFAR10 dataset  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

image_width = c10_image_width
image_height = c10_image_height
image_depth = c10_image_depth
num_labels = c10_num_labels 
batch_size = 10 # I just created this

#the dataset
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10 
test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10


#number of iterations and learning rate
num_steps = 101  #10001
display_step = 10  #1000
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
    bias = tf.Variable(tf.ones([num_labels]), tf.float32)
  
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
    print('Cifar10 training Initialized')
    for step in range(num_steps):
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        #_, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % display_step == 0):
            train_accuracy = accuracy(predictions, train_labels[:, :])
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)


#+++++++++++++++++++++ leNet5 on mnist +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

LENET5_BATCH_SIZE = 32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84
 
def variables_lenet5(patch_size = LENET5_PATCH_SIZE, patch_depth1 = LENET5_PATCH_DEPTH_1, 
                     patch_depth2 = LENET5_PATCH_DEPTH_2, 
                     num_hidden1 = LENET5_NUM_HIDDEN_1, num_hidden2 = LENET5_NUM_HIDDEN_2,
                     image_depth = 1, num_labels = 10):
    
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
 
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))
 
    w3 = tf.Variable(tf.truncated_normal([5*5*patch_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
 
    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables
 
def model_lenet5(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)

    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits


#parameters determining the model size
#image_size = mnist_image_size
#num_labels = mnist_num_labels

image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels 
batch_size = 10 # I just created this


#the datasets
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels
 
#number of iterations and learning rate
num_steps = 20
display_step = 5
learning_rate = 0.5 #0.001


graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5(image_depth = image_depth, num_labels = num_labels)
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5
    logits = model(tf_train_dataset, variables)
 
    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)

#+++++++++++++++++++++ leNet5 on cifar10 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
LENET5_BATCH_SIZE = 36 #32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84
 
def variables_lenet5(patch_size = LENET5_PATCH_SIZE, patch_depth1 = LENET5_PATCH_DEPTH_1, 
                     patch_depth2 = LENET5_PATCH_DEPTH_2, 
                     num_hidden1 = LENET5_NUM_HIDDEN_1, num_hidden2 = LENET5_NUM_HIDDEN_2,
                     image_depth = 1, num_labels = 10):
    
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
 
    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))
 
    w3 = tf.Variable(tf.truncated_normal([5*5*patch_depth2, num_hidden1], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
 
    w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
    
    w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return variables


def model_lenet5(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)

    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits


#parameters determining the model size

image_width = c10_image_width
image_height = c10_image_height
image_depth = c10_image_depth
num_labels = c10_num_labels 
batch_size = 10 # I just created this

#the dataset
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10 
test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10

#number of iterations and learning rate
num_steps = 101
display_step = 10
learning_rate = 0.5 #0.001

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5(image_depth = image_depth, num_labels = num_labels)
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5
    logits = model(tf_train_dataset, variables)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)

'''
#+++++++++++++++++++++ leNet5 on cifar10 using API +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#LENET5_BATCH_SIZE = 36 #32
#LENET5_PATCH_SIZE = 5
#LENET5_PATCH_DEPTH_1 = 6
#LENET5_PATCH_DEPTH_2 = 16
#LENET5_NUM_HIDDEN_1 = 120
#LENET5_NUM_HIDDEN_2 = 84


from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import fully_connected, dropout, flatten

def model_lenet5(data):
    layer1_conv = conv_2d(data, nb_filter = 6, filter_size = 5, strides = [1,1,1,1], activation='relu', padding='SAME', bias = True)
    layer1_pool = avg_pool_2d(layer1_conv, kernel_size = 2, strides=2, padding='SAME')

    layer2_conv = conv_2d(layer1_pool, nb_filter = 16, filter_size = 5, strides = [1,1,1,1], activation='relu', padding='VALID', bias = True)
    layer2_pool = avg_pool_2d(layer2_conv, kernel_size = 2, strides=2, padding='SAME')

    flat_layer = flatten(layer2_pool)
    layer3_fccd = fully_connected(flat_layer, n_units = 120, activation='relu', bias = True)

    layer4_fccd = fully_connected(layer3_fccd, n_units = 84, activation='relu', bias = True)

    #w = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
    #b = tf.Variable(tf.constant(1.0, shape = [10]))

    #logits = tf.matmul(layer4_fccd, w) + b
    logits = fully_connected(layer3_fccd, n_units = 10, activation='relu', bias = True)
    return logits


#parameters determining the model size

image_width = c10_image_width
image_height = c10_image_height
image_depth = c10_image_depth
num_labels = c10_num_labels 
batch_size = 10 # I just created this

#the dataset
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10 
test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10

#number of iterations and learning rate
num_steps = 20
display_step = 5
learning_rate = 0.5 #0.001

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5(image_depth = image_depth, num_labels = num_labels)
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5
    logits = model(tf_train_dataset)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('leNet5 via API - Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)


#+++++++++++++++++++++++++++++++++ AlexNet API on oxflower17 dataset +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import fully_connected, dropout, flatten
from tflearn.layers.normalization import batch_normalization 

ox17_image_width = 224
ox17_image_height = 224
ox17_image_depth = 3
ox17_num_labels = 17
 
m = 100 # 1000
import tflearn.datasets.oxflower17 as oxflower17
train_dataset_, train_labels_ = oxflower17.load_data(one_hot=True)
train_dataset_ox17, train_labels_ox17 = train_dataset_[:m,:,:,:], train_labels_[:m,:]
test_dataset_ox17, test_labels_ox17 = train_dataset_[m:,:,:,:], train_labels_[m:,:]
 
print('Training set', train_dataset_ox17.shape, train_labels_ox17.shape)
print('Test set', test_dataset_ox17.shape, test_labels_ox17.shape)


#parameters determining the model size

image_width = ox17_image_width
image_height = ox17_image_height
image_depth = ox17_image_depth
num_labels = ox17_num_labels 
batch_size = 10 

#the dataset
train_dataset = train_dataset_ox17
train_labels = train_labels_ox17 
test_dataset = test_dataset_ox17
test_labels = test_labels_ox17


#ALEX_PATCH_DEPTH_1, ALEX_PATCH_DEPTH_2, ALEX_PATCH_DEPTH_3, ALEX_PATCH_DEPTH_4 = 96, 256, 384, 256
#ALEX_PATCH_SIZE_1, ALEX_PATCH_SIZE_2, ALEX_PATCH_SIZE_3, ALEX_PATCH_SIZE_4 = 11, 5, 3, 3
#ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096

def model_alexnet(data):
    layer1_conv = conv_2d(data, nb_filter = 96, filter_size = 11, strides = [1,4,4,1], activation='relu', padding='SAME', bias = True)
    layer1_pool = avg_pool_2d(layer1_conv, kernel_size = 2, strides=2, padding='SAME')
    layer1_norm = batch_normalization(layer1_pool)
 
    layer2_conv = conv_2d(layer1_norm, nb_filter = 256, filter_size = 5, strides = [1,1,1,1], activation='relu', padding='SAME', bias = True)
    layer2_pool =  avg_pool_2d(layer2_conv, kernel_size = 2, strides=2, padding='SAME')
    layer2_norm = batch_normalization(layer2_pool)
 
    layer3_conv = conv_2d(layer2_norm, nb_filter = 384, filter_size = 3, strides = [1,1,1,1], activation='relu', padding='SAME', bias = True)
    layer3_relu = avg_pool_2d(layer3_conv, kernel_size = 2, strides=2, padding='SAME')
 
    layer4_conv = conv_2d(layer3_relu , nb_filter = 384, filter_size = 3, strides = [1,1,1,1], activation='relu', padding='SAME', bias = True)
    layer4_relu = avg_pool_2d(layer4_conv , kernel_size = 2, strides=2, padding='SAME')
 
    layer5_conv = conv_2d(layer4_relu, nb_filter = 256, filter_size = 5, strides = [1,1,1,1], activation='relu', padding='SAME', bias = True)
    layer5_pool = avg_pool_2d(layer5_conv , kernel_size = 2, strides=2, padding='SAME')
    layer5_norm = batch_normalization(layer5_pool)
 
    flat_layer = flatten(layer5_norm)
    layer6_fccd = fully_connected(flat_layer, n_units = 120, activation='relu', bias = True)
    layer6_drop = dropout(layer6_fccd, 0.5)
 
    layer7_fccd = fully_connected(flat_layer, n_units = 120, activation='relu', bias = True)
    layer7_drop = tf.nn.dropout(layer7_fccd, 0.5)
 
    w = tf.Variable(tf.truncated_normal([120, num_labels], stddev=0.1))
    b = tf.Variable(tf.constant(1.0, shape = [num_labels]))

    logits = tf.matmul(layer7_drop, w8) + b8
    return logits




#number of iterations and learning rate
num_steps = 20
display_step = 5
learning_rate = 0.5 #0.001

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5(image_depth = image_depth, num_labels = num_labels)
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_alexnet
    logits = model(tf_train_dataset)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('AlexNet oxflower dataset via API - Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)

'''

