from metric import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

outfiles = ['results-KNN.txt', 'results-SVM.txt', 'results-NN.txt']
'''
processes output file for knn.
input knn.txt
output {neighbors (int): [[accuracy, multiclass precision]]

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
    if filename == outfiles[0]:
        for k in range(0,15):
            if (atr == 'pred') or (atr == 'truth'):
                folds = [list(map(int, linelist[attrb[atr] + (55*k) + (i*10)].split(','))) for i in range(0,5)]
            else:
                folds = [linelist[attrb[atr] + (55*k) + (i*10)] for i in range(0,5)]
            data.append(folds)
    elif filename == outfiles[1]:
        for k in range(36):
            fold = []
            for i in range(5):
                ix = attrb[atr] + 55*k + i * 10
                current  = list(map(int, linelist[ix].split(',')))
                fold.append(current)
            data.append(fold)
    return data

'''
getMCC gets mcc of all data
input fold labels, fold predictions for all data
      param 1                                           param2
    [ [[labels fold 1] [labels fold 2] [labels fold 3]...] []]
    same input for predictions

output:
    list of:
        [[fold1mcc, fold2mcc ... fold k mcc]...[fold mcc for increment in parameter]]
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


def main():
    predictions =structData(outfiles[1], 'pred')
    labels = structData(outfiles[1], 'truth')
    cc = getconfusionMatrix(labels, predictions)
    # param
    allmcc = []
    prec = []
    rec = []
    for i in cc:
        # fold
        foldmcc = []
        precisions = []
        recalls = []
        for j in i:
            foldmcc.append(matthews(j))
            precision, recall = precisionrecall(j)
            # precision across all digits
            precisions.append(precision[1].mean())
            # recall across all digits
            recalls.append(recall[1].mean())
        prec.append(precisions)
        rec.append(recalls)
        allmcc.append(foldmcc)
    allmcc = getMCC(labels, predictions)
    fig, ax = plt.subplots(nrows=3, ncols = 1, figsize =(9,4))
    bpl = ax[0].boxplot(allmcc, vert = True)
    bpl2 = ax[1].boxplot(prec, vert = True)
    bpl2 = ax[2].boxplot(rec, vert = True)
    plt.show()
# main()
