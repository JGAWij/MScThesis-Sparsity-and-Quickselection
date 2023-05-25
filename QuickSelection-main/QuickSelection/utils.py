

"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
import numpy as np   
from sklearn import preprocessing  
import scipy
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split
import urllib.request as urllib2 
import errno
import os
import numpy as np
import sys; sys.path.append(os.getcwd())  
from Gini import gini_algorithm

"""*************************************************************************"""
"""                             Load data                                   """
    
def load_data(name):

    if name == "coil20":
        mat = scipy.io.loadmat('./datasets/COIL20.mat')
        X = mat['fea']
        y = mat['gnd'] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.StandardScaler().fit(X_train)
        
    elif name=="madelon":
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        X_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        X_test =  np.loadtxt(urllib2.urlopen(val_data_url))
        y_test =  np.loadtxt(urllib2.urlopen(val_resp_url))
        scaler = preprocessing.StandardScaler().fit(X_train)

        
    elif name=="har":         
        X_train = np.loadtxt('./datasets/UCI HAR Dataset/train/X_train.txt')
        y_train = np.loadtxt('./datasets/UCI HAR Dataset/train/y_train.txt')
        X_test =  np.loadtxt('./datasets/UCI HAR Dataset/test/X_test.txt')
        y_test =  np.loadtxt('./datasets/UCI HAR Dataset/test/y_test.txt')
        scaler = preprocessing.StandardScaler().fit(X_train)
       
    elif name == "MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        #The reshaping operation converts the images from a grid of pixels to a vector representation
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')
        print("x_test", X_test)
        print("x_train", X_train)
        print("y_test", y_test)
        print("y_train", y_train)
        #This scaler object is used to standardize the data based on the mean and standard deviation of the training set.
        scaler = preprocessing.StandardScaler().fit(X_train)

    elif name=="SMK":
        mat = scipy.io.loadmat('./datasets/SMK_CAN_187.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    
    elif name=="GLA":
        mat = scipy.io.loadmat('./datasets/GLA-BRA-180.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.StandardScaler().fit(X_train)
    elif name=="mac":
        mat = scipy.io.loadmat('./datasets/PCMAC.mat', squeeze_me=True)
        X = mat["X"]
        y = mat["Y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    return X_train, y_train, X_test, y_test
    
def check_path(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise 


def imbalanced_data(X_train, y_train, X_test, y_test, distribution, max_iterations=10):
    # Get the number of classes
    num_classes = len(distribution)

    #--------------------- Training set ---------------------
    # Create a dictionary to store the indices of instances for each class
    class_indices_y_train = {class_label: np.where(y_train == class_label)[0] for class_label in np.unique(y_train)}
    print("class_indices_y_train", class_indices_y_train)
    # Get the count of instances in each class based on the desired distribution
    original_counts_y_train = np.array([len(class_indices_y_train[class_label]) for class_label in np.unique(y_train)])
    # Initialize the adjusted counts with the original counts
    adjusted_counts_y_train = original_counts_y_train.copy()
    print("original_counts_y_train", adjusted_counts_y_train)
    # Calculate the target counts based on the desired distribution
    target_counts_y_train = np.round(distribution * np.sum(original_counts_y_train)).astype(int)
    # Limit the target counts to not exceed the original counts
    target_counts_y_train = np.minimum(target_counts_y_train, original_counts_y_train)
    print("target_counts _y_train", target_counts_y_train)

    # --------------------- Test set ---------------------
    # Create a dictionary to store the indices of instances for each class
    class_indices_y_test = {class_label: np.where(y_test == class_label)[0] for class_label in np.unique(y_test)}
    print("class_indices_y_test", class_indices_y_test)
    # Get the count of instances in each class based on the desired distribution
    original_counts_y_test = np.array([len(class_indices_y_test[class_label]) for class_label in np.unique(y_test)])
    # Initialize the adjusted counts with the original counts
    adjusted_counts_y_test = original_counts_y_test.copy()
    print("original_counts_y_test", adjusted_counts_y_test)
    # Calculate the target counts based on the desired distribution
    target_counts_y_test = np.round(distribution * np.sum(original_counts_y_test)).astype(int)
    # Limit the target counts to not exceed the original counts
    target_counts_y_test = np.minimum(target_counts_y_test, original_counts_y_test)
    print("target_counts_y_test", target_counts_y_test)

    # Create new empty lists to store the adjusted data
    adjusted_X_train = []
    adjusted_y_train = []

    # --------------------- First adjustment training data ---------------------
    for class_label in np.unique(y_train):
        print("class_label", class_label)
        class_indices_for_label_y_train = class_indices_y_train[class_label]
        print("class_indices_for_label y_train", class_indices_for_label_y_train)
        class_indices_sampled_y_train = np.random.choice(class_indices_for_label_y_train,
                                                         size=target_counts_y_train[class_label], replace=False)
        print("class_indices_sampled y_train", class_indices_sampled_y_train)
        adjusted_X_train.extend(X_train[class_indices_sampled_y_train])
        adjusted_y_train.extend(y_train[class_indices_sampled_y_train])

        # Update the adjusted counts based on the adjusted dataset
        print("len(class_indices_y_train[class_label]", len(class_indices_y_train[class_label]))
        adjusted_counts_y_train = np.array(
            [len([label for label in adjusted_y_train if label == class_label]) for class_label in
             np.unique(adjusted_y_train)])
        print("adjusted_counts_y_train", adjusted_counts_y_train)

        #adjusted_X_train = np.array(adjusted_X_train)
        #adjusted_y_train = np.array(adjusted_y_train)
        x = (adjusted_counts_y_train / np.sum(adjusted_counts_y_train))
        print("adjusted_counts_y_train / np.sum(adjusted_counts_y_train)", x)
        print("distribution", distribution)
        # Create new empty lists to store the adjusted data during each iteration
        adjusted_X_train_iteration = []
        adjusted_y_train_iteration = []

    # --------------------- Iteration adjustment training data ---------------------
    # Iteratively adjust the dataset until the desired distribution is reached or the maximum number of iterations is reached
    iteration = 0
    while not np.allclose(adjusted_counts_y_train / np.sum(adjusted_counts_y_train), distribution,
                              atol=1e-2) and iteration < max_iterations:

            print("WHILE LOOP STARTED FOR TRAINING DATA")
            class_indices_y_train = {class_label: np.where(adjusted_y_train == class_label)[0] for class_label in
                                     np.unique(adjusted_y_train)}
            target_counts_y_train = np.round(distribution * np.sum(adjusted_counts_y_train)).astype(int)
            # Limit the target counts to not exceed the original counts
            target_counts_y_train = np.minimum(target_counts_y_train, adjusted_counts_y_train)
            print("target_counts_y_train loop", target_counts_y_train)

            adjusted_X_train_iteration.clear()  # Clear the list for each iteration
            adjusted_y_train_iteration.clear()  # Clear the list for each iteration

            for class_label in np.unique(adjusted_y_train):
                print("class_label loop", class_label)
                class_indices_for_label_y_train = class_indices_y_train[class_label]
                print("class_indices_for_label y_train loop", class_indices_for_label_y_train)
                class_indices_sampled_y_train = np.random.choice(class_indices_for_label_y_train,
                                                                 size=target_counts_y_train[class_label], replace=False)
                print("class_indices_sampled y_train loop", class_indices_sampled_y_train)
                adjusted_X_train_iteration.extend(adjusted_X_train[class_indices_sampled_y_train])
                adjusted_y_train_iteration.extend(adjusted_y_train[class_indices_sampled_y_train])

            # Update the adjusted counts based on the adjusted dataset
            adjusted_counts_y_train = np.array(
                [len([label for label in adjusted_y_train_iteration if label == class_label]) for class_label in
                 np.unique(adjusted_y_train)])
            print("adjusted_counts_y_train", adjusted_counts_y_train)

            # Update the main adjusted data lists with the iteration-adjusted lists
            adjusted_X_train = adjusted_X_train_iteration.copy()
            adjusted_y_train = adjusted_y_train_iteration.copy()

            x = (adjusted_counts_y_train / np.sum(adjusted_counts_y_train))
            print("adjusted_counts_y_train / np.sum(adjusted_counts_y_train)", x)
            print("distribution", distribution)
            print("iterations", iteration)
            iteration += 1


    iteration2 = 0
    while not np.allclose(adjusted_counts_y_test / np.sum(adjusted_counts_y_test), distribution, atol=1e-2) and iteration2 < max_iterations:
        print("WHILE LOOP STARTED FOR TEST DATA")
        class_indices_y_test = {class_label: np.where(adjusted_y_test == class_label)[0] for class_label in
                                np.unique(adjusted_y_test)}
        target_counts_y_test = np.round(distribution * np.sum(adjusted_counts_y_test)).astype(int)
        # Limit the target counts to not exceed the original counts
        target_counts_y_test = np.minimum(target_counts_y_test, adjusted_counts_y_test)
        print("target_counts _y_test loop", target_counts_y_test)

        for class_label in np.unique(adjusted_y_test):
            print("class_label loop", class_label)
            class_indices_for_label_y_test = class_indices_y_test[class_label]
            print("class_indices_for_label y_test loop", class_indices_for_label_y_test)
            class_indices_sampled_y_test = np.random.choice(class_indices_for_label_y_test,
                                                            size=target_counts_y_test[class_label], replace=False)
            print("class_indices_sampled y_test loop", class_indices_sampled_y_test)
            adjusted_X_test.extend(adjusted_X_test[class_indices_sampled_y_test])
            adjusted_y_test.extend(adjusted_y_test[class_indices_sampled_y_test])

            # Update the adjusted counts based on the adjusted dataset
            print("len(class_indices_y_test[class_label] loop", len(class_indices_y_test[class_label]))
            adjusted_counts_y_test = np.array(
                [len([label for label in adjusted_y_test if label == class_label]) for class_label in
                 np.unique(adjusted_y_test)])
            print("adjusted_counts_y_test loop", adjusted_counts_y_test)

            x = (adjusted_counts_y_test / np.sum(adjusted_counts_y_test))
            print("adjusted_counts_y_test / np.sum(adjusted_counts_y_test)", x)
            print("distribution", distribution)

    return adjusted_X_train, adjusted_y_train, adjusted_X_test, adjusted_y_test






