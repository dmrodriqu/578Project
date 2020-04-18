import LoadData as ld
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd, FloatTensor
import numpy as np
use_cuda = False    # set to True if GPU available


#####################################################################################################################
"""
class Net: inherited from nn.Module
Defines a feed forward neural network
"""
#####################################################################################################################


class Net(nn.Module):
    """
    constructor
    Description: constructs a feed forward neural network
    Inputs: 2 inputs
        - h_sizes: describes the neural network structure. Type: List. E.g. [100, 50, 30, 10], defines
                  a neural network with three layers. The input layer has 100 nodes, first hidden layer has 50,
                  second hidden layer has 30 and the output layer has 10 nodes.
        - dropout: the dropout coefficient (to reduce overfitting). Type: float
    Outputs: N/A
    """
    def __init__(self, h_sizes, dropout):
        super(Net, self).__init__()
        self.layers = nn.Sequential()
        index = 0
        for k in range(len(h_sizes) - 2):
            self.layers.add_module(str(index), nn.Linear(h_sizes[k], h_sizes[k + 1]))
            index += 1
            self.layers.add_module(str(index), nn.ReLU())
            index += 1
            self.layers.add_module(str(index), nn.Dropout(p=dropout))
            index += 1
        self.layers.add_module(str(index), nn.Linear(h_sizes[-2], h_sizes[-1]))
        index += 1
        self.layers.add_module(str(index), nn.LogSoftmax(dim=1))


    """
    function forward(x)
    Description: computes the output of the neural network given the input vector
    Inputs: 1 input
        - x: the input vector. Type numpy vector
    Outputs: 1 output
        a numpy vector
    """
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

######################################################################################################################
"""
function create_output_data(labels)
Description: creates a matrix of one-hot encoded vectors for each entry in the labels
Inputs: 1 input
    - labels: a numpy array of labels. Each entry is an int between 0 to 9. Type: numpy vectory
Outputs: 1 output
    - a numpy matrix, whose rows are the same as the rows of labels, and columns are of size 10.
"""
######################################################################################################################


def create_output_data(labels, num_classes=10):
    n = labels.shape[0]
    output = np.zeros((n,num_classes))
    for i in range(n):
        vec = np.zeros(num_classes)     # a vector of zeros of size = number of classes
        vec[labels[i]] = 1
        output[i,:] = vec.transpose()
    return output
######################################################################################################################
"""
function train(nnet_sizes, train_data, train_labels)
Description: creates and trains a neural network
Inputs: 3 inputs
    - nnet_sizes  : a list containing number of neurons in each of the layer of the neural network. Type: list of ints
    - train_data  : the training data for the network to be trained over. Type: numpy matrix
    - train_labels: the ground truth for the training data. Type: numpy vector
Outputs: 1 output
    - trained neural network
"""
######################################################################################################################


def train(nnet_sizes, train_data, train_labels):
    dropout = 0.1
    device = torch.device("cude" if use_cuda else "cpu")
    model = Net(nnet_sizes, dropout).to(device)
    epochs = 25
    batch_size = 64
    learning_rate = 0.003
    weight_decay = 1e-3
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    x = autograd.Variable(FloatTensor(train_data))
    y = create_output_data(train_labels)
    y = FloatTensor(y)
    model.train()
    for epoch in range(epochs):
        losses = []
        for beg_i in range(0, x.size(0), batch_size):
            x_batch = x[beg_i:beg_i + batch_size, :]
            y_batch = y[beg_i:beg_i + batch_size, :]
            optimizer.zero_grad()
            logps = model(x_batch)    # log probabilities
            loss = criterion(logps, torch.max(y_batch, 1)[1])   # training loss
            loss.backward()
            optimizer.step()
            losses.append(loss.data.numpy())
        print(epoch,":" ,sum(losses)/float(len(losses)))
    return model


######################################################################################################################
"""
function nn_predict(model, data, labels)
Description: creates and trains a neural network
Inputs: 3 inputs
    - model : a trained neural network. Type Net
    - data  : the data for whom the labels are to be predicted using the model. Type: numpy matrix
    - labels: the ground truth for the data. Type: numpy vector
Outputs: 1 output
    - the predicted labels. Type: numpy vector of ints
"""
######################################################################################################################


def nn_predict(model, data, target):
    x = autograd.Variable(FloatTensor(data))
    y = FloatTensor(create_output_data(target))
    probs = model(x)  # predictions[i,:] is a vector of length 10 of probabilities.
    predicted_labels = torch.max(probs, 1)[1]
    return predicted_labels.numpy()



# def main():
#     data, val, datalabel, vallabel = ld.trainvalsplit('train')
#     nnet_sizes = [784, 128, 64, 10]
#     # model = Net(nnet_sizes, 0.05)
#     model = train(nnet_sizes, data, datalabel)
#     predictions = nn_predict(model, data, datalabel)
#     acc = 0
#     for i in range(len(datalabel)):
#         if (datalabel[i] == predictions[i]):
#             acc += 1
#     print(acc/float(len(datalabel)) * 100)
# main()

