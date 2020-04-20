import numpy as np
from collections import Counter
import random


'''
List: labels, List: predictions, optional (bool) -> 
    confusion matrix, rows true, columns predicted.
         ---+--- 
         |  |  |
    true +--+--+
         |  |  |
         +--+--+
        predicted
'''
def confusionMatrix(labels, predictions, normalized = False):
    uniquelab = set(labels)
    llab = len(uniquelab)
    cmat = np.zeros((llab,llab))
    counts = Counter(labels)
    for i in range(len(labels)):
        cmat[labels[i]-1, predictions[i]-1]+= 1
    if normalized:
        for i in range(len(cmat)):
            cmat[i] /= counts[i+1]
    return cmat

'''
generates precision and recall
input -> np.matrix ( confusion matrix)


output -> tuple -> np.array(size = n classes), np.array(size = n classes)
multiclass recision recall

'''
def precisionrecall(cmat):
    n = np.shape(cmat)[0]
    pm = np.zeros(n)
    rm = np.zeros(n)
    for i in range(len(pm)):
        # recall
        pm[i]=cmat[i,i]/sum(cmat[:,i])
        # precision
        rm[i]=cmat[i,i]/sum(cmat[i,:])
    return pm, rm

'''
accuracy: 
    calculates accuracy
    
input = list : labels, list : predictions
output = float : accuracy
'''

def accuracy(labels, predictions):
    n = len(labels)
    tot = 0
    for i in range(n):
        if labels[i] - predictions[i] == 0:
            tot += 1
    return tot/n

'''
calculates matthews correlation coefficient from confusion matris

input: np.mat(): confusion matrix
output: float


'''

def matthews(cmat):
    dim = np.shape(cmat)[0]
    trues =  sum([cmat[:,i] for i in range(dim)])
    preds = sum([cmat[i,:] for i in range(dim)])
    n= sum(preds)
    correct = sum([cmat[i,i] for i in range(dim)])
    cordiff  = correct * n - sum([trues[i]*preds[i] for i in range(dim)]) 
    cnormp = n ** 2 - sum([preds[i] ** 2 for i in range(len(preds))])
    cnormt = n ** 2 - sum([trues[i] ** 2 for i in range(len(trues))])
    res = cordiff/ (cnormp * cnormt)**(1/2)
    if np.isnan(res):
        return 0
    else:
        return res

