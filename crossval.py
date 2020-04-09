import LoadData as ld
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef 
from sklearn import svm
from math import ceil

def createix(folds):
    foldarr = []
    for i in range(folds):
        # create folds
        ix = [x%folds for x in range(i, i+folds)]
        loo = ix[0]
        fol = np.array(ix[1:])
        foldarr.append([loo,fol])
    return np.array(foldarr)

#######################################################################################################################
"""
function kfold(classifier, val, vallabel, folds)                                
description: 
    - runs kfold cross validation for a classifier. It splits the data into specified number of folds or subsets
      and trains of (folds - 1) of the subsets and tests on the remaining one subset. Each time the score on the
      test set is computed using the Mathews Correction Coefficient. 
Inputs: 4 inputs
    - classifier: a classifier e.g. SVM.                                                Type: sklearn classifier
    - val       : a matrix containing the validation data.                              Type numpy matrix
    - vallabel  : an array containing the true labels for the validation data (val).    Type: numpy array
    - k         : number of folds for k-fold cross validation.                          Type: integer

Outputs: 2 outputs          
    - mean of the scores for all the k-folds.       Type: numpy array
    - variance of the scores for all the k-folds.   Type: numpy array
"""
######################################################################################################################


def kfold(classifier, val, vallabel, k):
    # split the data into folds many disjoint sets #
    n = val.shape[0]    # the total number of data points
    subset_size = ceil(n / k)
    all_indices = set(range(n))
    scores = np.array(np.zeros(k))
    for i in range(k):
        print(i)
        validation_indices = set(range(subset_size*i,min(subset_size*(i+1), n)))  # the indices of the validation set
        train_indices = list(all_indices - validation_indices)              # train set is all data but the validation
        validation_indices = list(validation_indices)
        ##################################################################################
        # for test only, and shall be removed in the final version #
        training_data = val[train_indices]          # assemble the training dataset
        training_labels = vallabel[train_indices]
        validation_data = val[validation_indices]   # assemble the validation dataset
        validation_labels = vallabel[validation_indices]
        ###################################################################################
        # train the classifier on the training set #
        classifier.fit(val[train_indices], vallabel[train_indices])

        predictions = classifier.predict(val[validation_indices])
        confusion = confusion_matrix(vallabel[validation_indices], predictions)
        mcc = matthews_corrcoef(vallabel[validation_indices], predictions)
        # append mcc for evaluation
        scores[i] = mcc
        # store results

    return scores.mean(), scores.std()**2

# def kfold(classifier, val, vallabel, folds):
#     ix = np.arange(len(val))
#     splits = np.array_split(ix, folds)
#     foldix = createix(folds)
#     # for every fold
#     scores = np.array(np.zeros(folds))
#     for i in range(len(foldix)):
#         print(i)
#         # get the split ix
#         foldindices = foldix[i][1]
#         # get the indices of every datum in split
#         crossvalix = np.array(splits)[foldindices]
#         # train on this
#         validationData = val[np.concatenate(crossvalix)]
#         validationLabels = vallabel[np.concatenate(crossvalix)]
#         # test on this
#         leaveoneoutData = val[np.array(np.array(splits)[foldix[i][0]])]
#         leaveoneoutlabels = vallabel[np.array(np.array(splits)[foldix[i][0]])]
#         # call classifier
#         classifier.fit(validationData, validationLabels)
#         predictions = classifier.predict(leaveoneoutData)
#         confusion = confusion_matrix(leaveoneoutlabels, predictions)
#         mcc = matthews_corrcoef(leaveoneoutlabels, predictions)
#         # append mcc for evaluation
#         scores[i] = mcc
#         # store results
#     return scores.mean(), scores.std() **2


#######################################################################################################################
"""

"""
#######################################################################################################################
def crossValidation(classifier,validationData, validationLabels, folds):
    return kfold(classifier, validationData, validationLabels, folds)


def main():
    data, val, datalabel, vallabel = ld.trainvalsplit('train')
    svm_classifier = svm.SVC(gamma='auto')
    dummy_size = 10
    # td = data[:dummy_size]
    # tl = datalabel[:dummy_size]
    # print(kfold(svm_classifier,td,tl,5))
    print(kfold(svm_classifier, data, datalabel, 5))
main()