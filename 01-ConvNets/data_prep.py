import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as p

#from data_prep1 import prepare_data1

class prepare_data:
    #methods for one-hot encoding the labels, loading the data in a randomized array and a method for
    # flattening an array (since a fully connected network needs an flat array as its input):
    
    def randomize(self,dataset, labels):
        permutation = np.random.permutation(self.labels.shape[0])
        self.shuffled_dataset = self.dataset[permutation, :, :]
        self.shuffled_labels = self.labels[permutation]
        return self.shuffled_dataset, self.shuffled_labels
    
    def one_hot_encode(np_array):
        return (np.arange(10) == np_array[:,None]).astype(np.float32)
    
    def reformat_data(self,dataset, labels, image_width, image_height, image_depth):
        np_dataset_ = np.array([np.array(image_data).reshape(self.image_width, self.image_height, self.image_depth) for image_data in self.dataset])
        np_labels_ = one_hot_encode(np.array(self.labels, dtype=np.float32))
        self.np_dataset, self.np_labels = randomize(np_dataset_, np_labels_)
        return self.np_dataset, self.np_labels
    
    def flatten_tf_array(array):
        shape = array.get_shape().as_list()
        return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
    
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    
    #def __init__(self,df,train_dataset, train_labels,test_dataset,test_labels ):
    def __init__(self, df,train_dataset, train_labels,test_dataset,test_labels ):
        #load the MNIST and  CIFAR-10 datasets 
        self.df = df
        if(self.df == "mnist"):
            import mnist
            from mnist import MNIST
            mndata = MNIST('/home/terrence/CODING/Python/MODELS/CapsNets/python-mnist/data')

            mnist_image_width = 28
            mnist_image_height = 28
            mnist_image_depth = 1
            mnist_num_labels = 10

            self.train_dataset, self.train_labels = mndata.load_training()
            self.test_dataset, self.test_labels = mndata.load_testing()

            mnist_train_dataset, mnist_train_labels = prepare_data.reformat_data(self.train_dataset, self.train_labels, mnist_image_width, mnist_image_height, mnist_image_depth)
            mnist_test_dataset, mnist_test_labels = prepare_data.reformat_data(self.test_dataset, self.test_labels, mnist_image_width, mnist_image_height, mnist_image_depth)
            '''
            print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
            print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_width*mnist_image_height*1))
            print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))

            print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
            print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
    
            train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
            test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
            '''
            self.train_dataset = mnist_train_dataset
            self.train_labels = mnist_train_labels
            self.test_dataset = mnist_test_dataset
            self.test_labels = mnist_test_labels
            ######################################################################################
        #print("+++ cifar10 ++++++++++")
        elif(self.df == cifar10):
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
            '''
            print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
            print('Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
            print('Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)
            '''
            self.train_dataset = train_dataset_cifar10
            self.train_labels = train_labels_cifar10
            self.test_dataset = test_dataset_cifar10
            self.test_labels = test_labels_cifar10

    def _display(self):
        return self.train_dataset, self.train_labels,  self.test_dataset, self.test_labels         

