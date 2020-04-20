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

