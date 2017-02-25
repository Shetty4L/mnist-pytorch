from __future__ import print_function
import os
import struct
import gzip
import sys
import numpy as np
import torch
import torch.utils.data as data_utils

def save_mnist_data_to_file():
    
    if not os.path.exists("data/train_images.gz") and not os.path.exists("data/train_labels.gz") and not os.path.exists("data/test_images.gz") and not os.path.exists("data/test_labels.gz"):
        print("Processing training images: ")
        with gzip.open('data/train-images-idx3-ubyte.gz') as f:
            magic_num = struct.unpack(">i", f.read(4))[0]
            num_images = struct.unpack(">i", f.read(4))[0]
            height = struct.unpack(">i", f.read(4))[0]
            width = struct.unpack(">i", f.read(4))[0]
            train_images = []
        
            i = 1
            image = []
            while True:
                byte = f.read(1)
                if len(byte) == 0:
                    break
                image.append(struct.unpack(">B", byte)[0])
                if i % (width*height) == 0:
                    sys.stdout.flush()
                    sys.stdout.write("{0}/{1} training images processed\r".format(i / (width*height), num_images))
                    train_images.append(np.array(image))
                    image = []
        
                i += 1
        
        print("\nProcessing training labels: ")
        with gzip.open('data/train-labels-idx1-ubyte.gz') as f:
            magic_num = struct.unpack(">i", f.read(4))[0]
            num_images = struct.unpack(">i", f.read(4))[0]
            train_labels = []
        
            i = 1
            while True:
                byte = f.read(1)
                if len(byte) == 0:
                    break
                label = struct.unpack(">b", byte)[0]
                sys.stdout.flush()
                sys.stdout.write("{0}/{1} training labels processed\r".format(i, num_images))
                train_labels.append(label)
                i += 1
        
        print("\nProcessing test images: ")
        with gzip.open('data/t10k-images-idx3-ubyte.gz') as f:
            magic_num = struct.unpack(">i", f.read(4))[0]
            num_images = struct.unpack(">i", f.read(4))[0]
            height = struct.unpack(">i", f.read(4))[0]
            width = struct.unpack(">i", f.read(4))[0]
            test_images = []
        
            i = 1
            image = []
            while True:
                byte = f.read(1)
                if len(byte) == 0:
                    break
                image.append(struct.unpack(">B", byte)[0])
                if i % (width*height) == 0:
                    sys.stdout.flush()
                    sys.stdout.write("{0}/{1} testing images processed\r".format(i / (width*height), num_images))
                    test_images.append(np.array(image))
                    image = []
                i += 1
        
        print("\nProcessing test labels: ")
        with gzip.open('data/t10k-labels-idx1-ubyte.gz') as f:
            magic_num = struct.unpack(">i", f.read(4))[0]
            num_images = struct.unpack(">i", f.read(4))[0]
            test_labels = []
        
            i = 1
            while True:
                byte = f.read(1)
                if len(byte) == 0:
                    break
                label = struct.unpack(">b", byte)[0]
                sys.stdout.flush()
                sys.stdout.write("{0}/{1} testing labels processed\r".format(i, num_images))
                test_labels.append(label)
                i += 1
                
        print("\nSaving to file: ")
        np.save("data/train_images", np.array(train_images))
        np.save("data/train_labels", np.array(train_labels))
        np.save("data/test_labels", np.array(test_labels))
        np.save("data/test_images", np.array(test_images))
        print("Files saved.")
    else:
        print("Files already exist.")

def get_mnist_data():
    print("Loading data: ")
    train_images = np.load('data/train_images.npy')
    train_labels = np.load('data/train_labels.npy')
    test_images = np.load('data/test_images.npy')
    test_labels = np.load('data/test_labels.npy')
    print("Data loaded.")
    
    return train_images, train_labels, test_images, test_labels

def get_mnist_trainloader(batch_size=128):
    print("Loading data: ")
    train_images = torch.from_numpy(np.loadtxt('data/train_images.gz'))
    train_labels = torch.from_numpy(np.loadtxt('data/train_labels.gz'))
    train = data_utils.TensorDataset(train_images, train_labels)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    print("Training iterator created.")
    return train_loader

def preprocess(val_split=0.8):
    train_images, train_labels, test_images, test_labels = get_mnist_data()
    
    num_train = int(val_split*train_images.shape[0])
    num_val = int((1-val_split)*train_images.shape[0])
    num_test = test_images.shape[0]
    
    val_images = train_images[-(num_val+1):]
    val_labels = train_labels[-(num_val+1):]
    
    train_images = train_images[:num_train]
    train_labels = train_labels[:num_train]
    
    mean = np.mean(train_images, axis=0)
    std = np.std(train_images, axis=0)
    epsilon = 1e-6
    
    train_images = train_images.astype(float) - mean
    train_images /= (std + epsilon)
    
    val_images = np.subtract(val_images, mean)
    val_images /= (std + epsilon)
    
    test_images = np.subtract(test_images, mean)
    test_images /= (std + epsilon)
    
    class Data():
        def __init__(self, train_images, train_labels, val_images, val_labels, test_images, test_labels):
            self.train_images = train_images
            self.train_labels = train_labels
            self.val_images = val_images
            self.val_labels = val_labels
            self.test_images = test_images
            self.test_labels = test_labels
    
    data = Data(train_images, train_labels, val_images, val_labels, test_images, test_labels)
    return data
