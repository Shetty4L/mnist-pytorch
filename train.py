from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import torch
from torch.autograd import Variable

def model(input_size, hidden_size, output_size, learning_rate=0.001, regularization=0):
    print("Initializing model")
    net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(hidden_size, output_size))
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=regularization)
    print("Model initialized.")
    
    return net, loss_fn, optimizer
    
def train(data, net, loss_fn, optimizer, batch_size=256, num_iters=2000):
    print("Begin training: ")
    X_train = data.train_images
    y_train = data.train_labels
    losses = []
    training_acc = []
    val_acc = []
    test_acc = []
    
    for i in xrange(num_iters):
        indices = np.random.choice(X_train.shape[0], batch_size, replace=True)
        X_batch = Variable(torch.from_numpy(X_train[indices]).float())
        y_batch = Variable(torch.from_numpy(y_train[indices]), requires_grad=False)
        
        optimizer.zero_grad()
        y = net(X_batch)
        loss = loss_fn(y, y_batch)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data[0])
        
        if i % 100 == 0 or i == num_iters-1:
            print("[{}] loss: {}".format(i, loss.data[0]))
            
        if i % 500 == 0 or i == num_iters-1:
            accuracy = evaluate(X_batch, y_batch, net, mode="train")
            training_acc.append(accuracy)
            
            accuracy = evaluate(data.val_images, data.val_labels, net, mode="validation")
            val_acc.append(accuracy)
                
            accuracy = evaluate(data.test_images, data.test_labels, net, mode="test")
            test_acc.append(accuracy)
    
    accuracies = (training_acc, val_acc, test_acc)
    print("Training finished.")
    return losses, accuracies
    
def evaluate(X, y, net, mode="train"):
    print("Beginning evaluation:")
    correct = 0
    
    if mode == "train":
        total = X.size(0)
    else:
        total = X.shape[0]
        X = Variable(torch.from_numpy(X).float())
        y = Variable(torch.from_numpy(y))
    
    y_pred = net(X)
    _, predicted = torch.max(y_pred.data, 1)
    correct = (y.data == predicted).sum()
    
    accuracy = float(correct) / total
    print("Accuracy of the network of {} {} images: {:.3f}%".format(total, mode, 100*accuracy))
    return accuracy

def main():
    data = utils.preprocess()
    
    net, loss_fn, optimizer = model(data.train_images.shape[1], 1000, 10, learning_rate=0.001, regularization=0.0005)
    
    num_iters = 10000
    losses, accuracies = train(data, net, loss_fn, optimizer, num_iters=num_iters)
    training_acc, val_acc, test_acc = accuracies
    
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.plot(losses)
    plt.show()
    
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    training_acc_legend, = plt.plot([i for i in range(0, num_iters+1, 500)], training_acc, color='r', label="Training Accuracy")
    val_acc_legend, = plt.plot([i for i in range(0, num_iters+1, 500)], val_acc, color='b', label="Validation Accuracy")
    test_acc_legend, = plt.plot([i for i in range(0, num_iters+1, 500)], test_acc, color='k', label="Testing Accuracy")
    plt.legend(handles=[training_acc_legend, val_acc_legend, test_acc_legend])
    plt.show()

    
if __name__ == '__main__':
    main()
    
    
