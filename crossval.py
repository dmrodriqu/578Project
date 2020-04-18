import LoadData as ld
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from math import ceil
from sklearn.svm import SVC, LinearSVC
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import pandas as pd
import nnet
import time


def createix(folds):
    foldarr = []
    for i in range(folds):
        # create folds
        ix = [x%folds for x in range(i, i+folds)]
        loo = ix[0]
        fol = np.array(ix[1:])
        foldarr.append([loo,fol])
    return np.array(foldarr)


def performance(actual_labels, predicted_labels, classes):
    """
    takes the lists of predicted and actual labels for the whole dataset, along with the list of class labels
    and produces the measures of performance: true positives (TP), true negatives (TN), false positives (FP),
    and false negatives (FN)

    :param actual_labels: actual labels for each data instance. Type: numpy vector
    :param predicted_labels: predicted labels for each data instance. Type: numpy vector
    :param classes: a list of all classes. Type: numpy vector
    :return: a dictionary with keys class labels, and values a list of four numbers: TP, FP, TN, and FN
             e.g. {0: [5, 10, 2, 1], ..}
    """
    try:
        assert actual_labels.shape == predicted_labels.shape
    except:
        print('the sizes of actual labels and predictions do not match')
        return

    measures = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'count': 0}
    scores = {}
    for k in classes:
        scores[k] = measures.copy()

    for k in classes:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(predicted_labels.shape[0]):
            if predicted_labels[i] == k:   # this is a positive prediction for k-th class
                if actual_labels[i] == k:
                    tp += 1
                else: # predictions is k, actual is not k. Hence false positive
                    fp += 1
            else:   # this is a negative prediction for k-th class
                if actual_labels[i] == k:   # this instance is indeed of class k, but the prediction is not k. Hence FN
                    fn += 1
                else:   # prediction is not k, actual is not k. Hence this is a true negative
                    tn += 1
        scores[k]['TP'] = tp
        scores[k]['FP'] = fp
        scores[k]['TN'] = tn
        scores[k]['FN'] = fn
    class_label, counts = np.unique(actual_labels, return_counts=True)

    for i in range(len(class_label)):
        scores[class_label[i]]['count'] = counts[i]

    return scores


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


def kfold(classifier, nnet_sizes, data, labels, k):
    # split the data into folds many disjoint sets #
    n = data.shape[0]    # the total number of data points
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
        training_data = data[train_indices]          # assemble the training dataset
        training_labels = labels[train_indices]
        validation_data = data[validation_indices]   # assemble the validation dataset
        validation_labels = labels[validation_indices]
        ###################################################################################
        # train the classifier on the training set #
        if not nnet_sizes:  # the classifier is not a neural network
            classifier.fit(data[train_indices], labels[train_indices])
            predictions = classifier.predict(data[validation_indices])
        else:                   # the classifier is a neural network
            model = nnet.train(nnet_sizes, data[train_indices], labels[train_indices])
            predictions = nnet.nn_predict(model, data[validation_indices], labels[validation_indices])


        confusion = confusion_matrix(labels[validation_indices], predictions)
        mcc = matthews_corrcoef(labels[validation_indices], predictions)
        # append mcc for evaluation
        scores[i] = mcc
        # print(scores)
        # store results

    return scores

'''
def kfold(classifier, val, vallabel, folds):
     ix = np.arange(len(val))
     splits = np.array_split(ix, folds)
     foldix = createix(folds)
     # for every fold
     scores = np.array(np.zeros(folds))
     print(scores)
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
         print(mcc)
         # append mcc for evaluation
         scores[i] = mcc
         # store results
     return scores.mean(), scores.std() **2

'''

#######################################################################################################################
"""

"""
#######################################################################################################################
def crossValidation(classifier,validationData, validationLabels, folds):
    return kfold(classifier, validationData, validationLabels, folds)


#######################################################################################################################
"""
function anova(results)
description:
    - uses the results of k-fold cross validation to run a one-way ANOVA to determine if the mean of all of 
      the different populations is the same (null hypothesis) or not
Inputs: 1 input
    - results: a matrix where each row represents the data for one population. Type: list of (lists of numbers)
Outputs: 1 output
    - True if null hypothesis is NOT rejected
    - False, if null hypothesis is rejected
"""
#######################################################################################################################


def anova(results):
    f, p = stats.f_oneway(*results)
    return p < 5e-2


#######################################################################################################################
"""
function tukeyhsd(results)
description:
    - runs the Tukey HSD test on the results to find which of the population mean is statistically significantly 
      different from the others
Inputs: 1 input
    - results: a matrix where each row represents the data for one population. Type: list of (list of numbers)
Outputs: 1 output
    - a list of lists, where each small list has two integers, i and j, which indicate that classifier i
      and classifier j have means which are statistically significantly different
"""
#######################################################################################################################


def tukeyhsd(results):
    # create a pandas DataFrame from results #
    df = pd.DataFrame()
    for i in range(len(results)):
        df[str(i)] = results[i]
    stacked_data = df.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id', 'level_1': 'classifier', 0: 'result'})

    multicomp = MultiComparison(stacked_data['result'], stacked_data['classifier'])
    tukey_analysis = multicomp.tukeyhsd()
    sig_diffs = tukey_analysis.reject
    n = len(results) # number of classifier settings
    ind = 0
    sig_diff_classifiers = []
    for i in range(0,n-1):
        for j in range(i+1,n):
            if (sig_diffs[ind]):  # the means of classifier i and classifier j are statistically significantly different
                sig_diff_classifiers.append([i,j])
            ind += 1
    return sig_diff_classifiers



def run():
    data, val, datalabel, vallabel = ld.trainvalsplit('train')
    cf1 = SVC(C = 1, kernel = 'rbf', gamma='auto')
    cf2 = SVC(C=10, kernel='rbf', gamma='auto')
    # cf2 = LinearSVC(random_state=0, tol=1e-5)
    cf3 = SVC(C=100, kernel='rbf', gamma='auto')
    cf4 = SVC(C=500, kernel='rbf', gamma='auto')
    cf5 = SVC(C=1000, kernel='rbf', gamma='auto')
    nb_classes = 10
    dummy_size = 1000
    td = data[:dummy_size]
    tl = datalabel[:dummy_size]
    # print(tl)
    # s1 = kfold(cf1,[784, 128, 64, 10],td,tl,5)
    # start = time.time()
    # s2 = kfold(cf2,[],data,datalabel,5)
    # print((time.time() - start)/60.0)
    # s3 = kfold(cf3,td,tl,5)
    # s4 = kfold(cf4, td, tl, 5)
    # s5 = kfold(cf5, td, tl, 5)
    #
    # results = [s1,s2,s3,s4,s5]
    #
    # if anova(results):
    #     res = tukeyhsd(results)
    #
    # print(res)

    # pred_labels = np.random.randint(0,10,datalabel.shape[0])
    # classes = list(range(10))
    actual_labels = [0, 0, 1, 2]
    pred_labels = [1, 1, 1, 2]
    classes = [0, 1, 2]
    performance(np.array(actual_labels), np.array(pred_labels), classes)


if __name__ == '__main__':
    run()
