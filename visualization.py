from metric import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

outfiles = ['results-KNN.txt', 'results-SVM.txt', 'results-NN.txt']
'''
processes output file for knn.
input knn.txt
output {neighbors (int): [[accuracy, multiclass precision]]
'''

def processknn():
    testfile = outfiles[0]
    f = open(testfile, 'r')
    predlab = []
    a = ''
    b = ''
    cluster = 0
    iters = {}
    currclust = []
    for line in f:
        if 'labels:' in line:
            a = f.readline()
            a  = list(map(int,a.split(',')))
            acc = accuracy(a,b)
            pr = precisionrecall(confusionMatrix(a,b))[0]
            currclust.append([acc, pr])
        elif 'predictions' in line:
            b = f.readline()
            b= list(map(int, b.split(',')))
        elif 'Clusters' in line: 
            if cluster >  0:
                iters[cluster] = currclust
                currclust = []
            cluster += 1
    return iters


def boxplots(kn ,precision=False, digit = 0):
    alldat = []
    allpr = []
    for i in kn:
        accarr = []
        prarr = []
        if kn[i]:
            acc = kn[i]
            for j in acc:
                if not precision:
                    accarr.append(j[0])
                else:
                    prarr.append(j[1][digit])
        alldat.append(accarr)
        allpr.append(prarr)
    fig, ax = plt.subplots(nrows=1, ncols = 1, figsize =(9,4))
    if precision:
        bpl = ax.boxplot(allpr, vert = True)
    else:
         bpl = ax.boxplot(alldat, vert = True)
    plt.show()

kn = processknn()
boxplots(kn, precision = True, digit = 1)
