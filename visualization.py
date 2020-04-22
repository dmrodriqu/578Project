from metric import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from joblib import Parallel, delayed

outfiles = ['results-KNN.txt', 'results-SVM.txt', 'results-NN.txt']

'''
0 description
2 fold number
4 dictionary tp/fp
7 predictions
10 labels
difference between description = 55
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


def digitprec(cmat):
    digits = dict.fromkeys([x for x in range(10)], [])
    precision, recall = precisionrecall(cmat)
    for i in range(10):
        digits[i] = precision[i].mean()
    return digits

def plotting(digit, n):
    predictions =structData(outfiles[n], 'pred')
    labels = structData(outfiles[n], 'truth')
    cc = getconfusionMatrix(labels, predictions)
    # param
    allmcc = []
    prec = []
    rec = []
    digitdict = dict.fromkeys([x for x in range(10)], [])
    for i in cc:
        # fold
        foldmcc = []
        precisions = []
        recalls = []
        for j in i:
            precision, recall = precisionrecall(j)
            # precision across all digits
            precisions.append(precision[digit].mean())
            # recall across all digits
            recalls.append(recall[digit].mean())
        prec.append(precisions)
        rec.append(recalls)
        allmcc.append(foldmcc)
    allmcc = getMCC(labels, predictions)
    return allmcc, prec, rec

def plotall(alg , boxplot = 'precision'):
    mcs=[]
    precs = []
    recs = []
    tick  = [x for x in range(0, 36, 6)]
    tickdic = dict.fromkeys(tick)
    cs = [1, 2, 3, 4, 5, 6]
    poly  = [1, 2 , 3, 4 , 5]

    if alg == 'svm':
        n = 1
    elif alg == 'nn':
        n =2
    else:
        n = 0

    # create labels
    for i in range(0,36,6):
        tickdic[i] = 'c: {:1.1f}, deg [1,6]'.format(cs[i//6])
    ticks = ['{:}'.format(tickdic[i]) for i in tick ]
    # get all digit precision and recall
    for i in range(10):
        allmcc, prec, rec = plotting(i, n)
        mcs.append(allmcc)
        precs.append(prec)
        recs.append(rec)

    # create plot
    fig, ax = plt.subplots(nrows=6, ncols = 1, figsize =(10,10), tight_layout = True)
    bpl = ax[0].boxplot(allmcc, vert = True)

    # create all boxplots
    if boxplot  == 'precision':
        precs = precs
    else:
        precs = recs
    for i in range(len(precs)//2):
        ax[i+1].boxplot(precs[i], vert = True)
        ax[i+1].set_title('Digit {:} {:}'.format(i, boxplot))
        #bpl2 = ax[2].boxplot(rec, vert = True)
    if alg == 'svm':
        ax[-1].xaxis.set_ticks(tick)
        ax[-1].xaxis.set_ticklabels(ticks)
    ax[0].set_title("Matthews Correlation Coefficient (All Classes)")
    ax[3].set_ylabel(boxplot)
    plt.show()
    fig, ax = plt.subplots(nrows=6, ncols = 1, figsize =(10,10), tight_layout = True)
    bpl = ax[0].boxplot(allmcc, vert = True)
    for i in range(len(precs)//2, len(precs)):
        print(i)
        ax[i-len(precs)//2 + 1].boxplot(precs[i], vert = True)
        ax[i-len(precs)//2 + 1].set_title('Digit {:} {}'.format(i,boxplot))
        #bpl2 = ax[2].boxplot(rec, vert = True)
    if alg == 'svm':
        ax[-1].xaxis.set_ticks(tick)
        ax[-1].xaxis.set_ticklabels(ticks)
    ax[0].set_title("Matthews Correlation Coefficient (All Classes)")
    ax[3].set_ylabel(boxplot)
    plt.show()

plotall('knn', boxplot = 'recall')
plotall('svm', boxplot = 'precision')
