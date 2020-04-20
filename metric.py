import numpy as np
from collections import Counter

def confusionMatrix(labels, predictions, normalized = False):
    uniquelab = set(labels)
    llab = len(uniquelab)
    cmat = np.zeros((llab,llab))
    counts = Counter(labels)
    print(counts)
    for i in range(len(lab)):
        cmat[labels[i]-1, predictions[i]-1]+= 1
    if normalized:
        for i in range(len(cmat)):
            cmat[i] /= counts[i+1]
    return cmat

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

def accuracy(labels, predictions):
    n = len(labels)
    tot = 0
    for i in range(n):
        if lab[i] - pred[i] == 0:
            tot += 1
    print(tot/n)

