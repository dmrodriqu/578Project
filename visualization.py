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

'''
kn = processknn()
boxplots(kn, precision = True, digit = 1)
'''

'''
0 neighbors
2 fold number
4 dictionary tp/fp
7 predictions
10 labels
difference between  neighbors = 55
'''
def structData(filename, atr):
    attrb = {'neighbors': 0, 
            'fold': 2,
            'dict': 4,
            'pred': 7,
            'truth': 10}
    f = open(filename, 'r')
    s = f.read() 
    f.close()
    linelist = s.splitlines()
    #print(linelist[55])
    data  = []
    for k in range(0,15):
        if atr == 'pred' or 'truth':
            folds = [list(map(int, linelist[attrb[atr] + (55*k) + (i*10)].split(','))) for i in range(0,5)]
        else:
            folds = [linelist[attrb[atr] + (55*k) + (i*10)] for i in range(0,5)]
        data.append(folds)
    return data

'''
mcc per folds
'''
def getMCC(labels, predictions):
    predlen = len(predictions)
    return ([[accuracy(labels[i][k], predictions[i][k]) for k in range(len(predictions[i]))] for i in range(predlen)])


'''
confusion matrix per fold
'''
def getconfusionMatrix(labels, predictions):
    predlen = len(predictions)
    return ([[confusionMatrix(labels[i][k], predictions[i][k]) for k in range(len(predictions[i]))] for i in range(predlen)])


predictions =structData(outfiles[0], 'pred')
labels = structData(outfiles[0], 'truth')

cc = getconfusionMatrix(labels, predictions)
print(np.shape(cc[0][1]))

allmcc = getMCC(labels, predictions)
fig, ax = plt.subplots(nrows=1, ncols = 1, figsize =(9,4))
bpl = ax.boxplot(allmcc, vert = True)
plt.show()


    

