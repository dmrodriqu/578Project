import gzip
import io
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# finds relative path of data files
def createPath():
    datadir = './data/'
    datasets = os.listdir(datadir)
    datasets.sort()
    train= {}
    test = {}
    j = 0
    k = 0
    for i in range(len(datasets)):
        if 't10k' in datasets[i]:
            test[k] = datasets[i] 
            k += 1
        else:
            train[j] = datasets[i]
            j += 1
    return train,test

# converts compressed data to numpy array
def loadData(datadic):
    trainTest = {'data' : None, 'label': None}
    for i in range(0,2):
        filename = '{:}{:}'.format('./data/', datadic[i])
        f = gzip.open(filename, 'rb')
        if i == 0:
            trainTest['data'] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        else:
            trainTest['label'] = np.frombuffer(f.read(), np.uint8, offset=8)
        f.close()
    return trainTest

# splits data into train/val or returns test data
def trainvalsplit(trainortest):
    train, test = createPath()
    if trainortest == 'train':
        trainset = loadData(train)
        train = trainset['data']
        trainlabels = trainset['label']
        # returns xdata, valdata, labeldata, labelval
        return train_test_split(train,trainlabels, test_size = 0.2)
    else:
        return loadData(test)
