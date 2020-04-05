import LoadData as ld
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef 

def createix(folds):
    foldarr = []
    for i in range(folds):
        # create folds
        ix = [x%folds for x in range(i, i+folds)]
        loo = ix[0]
        fol = np.array(ix[1:])
        foldarr.append([loo,fol])
    return np.array(foldarr)


def kfold(classifier, val, vallabel, folds):
    ix = np.arange(len(val))
    splits = np.array_split(ix, folds)
    foldix = createix(folds)
    # for every fold
    scores = np.array(np.zeros(folds))
    for i in range(len(foldix)):
        # get the split ix
        foldindices = foldix[i][1]
        # get the indices of every datum in split
        crossvalix = np.array(splits)[foldindices]
        # train on this
        validationData = val[np.concatenate(crossvalix)]
        validationLabels = vallabel[np.concatenate(crossvalix)]
        # test on this
        leaveoneoutData = val[np.array(np.array(splits)[foldix[i][0]])]
        leaveoneoutlabels = vallabel[np.array(np.array(splits)[foldix[i][0]])]
        # call classifier 
        classifier.fit(validationData, validationLabels)
        predictions = classifier.predict(leaveoneoutData)
        confusion = confusion_matrix(leaveoneoutlabels, predictions)
        mcc = matthews_corrcoef(leaveoneoutlabels, predictions)
        # append mcc for evaluation
        scores[i] = mcc
        # store results
    return scores.mean(), scores.std() **2

def crossValidation(classifier,validationData, validationLabels, folds):
    return kfold(classifier, validationData, validationLabels, folds)
