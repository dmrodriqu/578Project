import LoadData as ld
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd, FloatTensor
import numpy as np
use_cuda = False    # set to True if GPU available


class Net(nn.Module):
    """
    class Net
    derives from PyTorch Module
    defines a feed forward artificial neural network (a multilayer perceptron)
    """

    def __init__(self, h_sizes, dropout):
        """
        constructor
        Description: constructs a feed forward neural network
        :param h_sizes: describes the neural network structure. Type: List. E.g. [100, 50, 30, 10], defines
                        a neural network with three layers. The input layer has 100 nodes, first hidden layer has 50,
                        second hidden layer has 30 and the output layer has 10 nodes.
        :param dropout: the dropout coefficient (to reduce overfitting). Type: float
        """
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

    def forward(self, x):
        """
        function forward(x)
        Description: computes the output of the neural network given the input vector
        :param x: the input vector. Type numpy vector
        :return: a numpy vector
        """
        for layer in self.layers:
            x = layer(x)
        return x


def create_output_data(labels, num_classes=10):
    """
    function create_output_data(labels)
    Description: creates a matrix of one-hot encoded vectors for each entry in the labels
    :param labels: labels: a numpy array of labels. Each entry is an int between 0 to 9. Type: numpy vector
    :param num_classes: number of actual classes
    :return: a numpy matrix, whose rows are the same as the rows of labels, and columns are of size num_classes.
    """
    n = labels.shape[0]
    output = np.zeros((n,num_classes))
    for i in range(n):
        vec = np.zeros(num_classes)     # a vector of zeros of size = number of classes
        vec[labels[i]] = 1
        output[i,:] = vec.transpose()
    return output


def train(nnet_sizes, train_data, train_labels):
    """
    function train(nnet_sizes, train_data, train_labels)
    Description: creates and trains a neural network
    :param nnet_sizes: a list containing number of neurons in each of the layer of the neural network. Type: list of ints
    :param train_data: the training data for the network to be trained over. Type: numpy matrix
    :param train_labels: the ground truth for the training data. Type: numpy vector
    :return: trained neural network
    """
    dropout = 0.05
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


def nn_predict(model, data):
    """
    function nn_predict(model, data, labels)
    Description: creates and trains a neural network
    :param model: a trained neural network. Type Net
    :param data: the data for whom the labels are to be predicted using the model. Type: numpy matrix
    :return: the predicted labels. Type: numpy vector of ints
    """
    x = autograd.Variable(FloatTensor(data))
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

