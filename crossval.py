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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network


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


def kfold(classifier, data, labels, num_classes, k):
    """
    function kfold(classifier, val, vallabel, folds)
    description: runs kfold cross validation for a classifier. It splits the data into specified number
    of folds or subsets and trains of (folds - 1) of the subsets and tests on the remaining one subset.
    Each time the score on the test set is computed using the Mathews Correction Coefficient.
    :param classifier: a classifier e.g. SVM. Type: sklearn classifier, or neural network
    :param data: a matrix containing the validation data. Type: numpy matrix
    :param labels: an array containing the true labels for the validation data (val). Type: numpy array
    :param num_classes: number of classes. Type: int
    :param k: number of folds for k-fold cross validation. Type: integer
    :return: for each fold, the function writes data to a text file containing the true positives, false positives,
             true negatives, and false negatives computed by the classifier after training for each class.
    """
    # split the data into folds many disjoint sets #
    n = data.shape[0]    # the total number of data points
    subset_size = ceil(n / k)
    all_indices = set(range(n))
    scores = np.array(np.zeros(k))
    classes = list(range(num_classes))
    #################################################################################################
    # CODE USED TO WRITE DATA TO TEXT FILES
    # ofile = open('results.txt', 'a+')
    # ofile.write("K-Nearest Neighbors. # of Clusters: " + str(classifier.n_neighbors) + '\n\n')
    # ofile.write("NNET Size: " + ','.join([str(i) for i in nnet_sizes]) + '\n\n')
    # ofile.write("C: " + str(classifier.C) + ", kernel: " + classifier.kernel
    #             + ', degree: ' + str(classifier.degree)
    #             + ', gamma: ' + str(classifier.gamma) + "\n\n")
    ###################################################################################################
    for i in range(k):
        print("fold #: ", i)
        # ofile.write('fold #: ' + str(i) + '\n\n')
        validation_indices = set(range(subset_size*i,min(subset_size*(i+1), n)))  # the indices of the validation set
        train_indices = list(all_indices - validation_indices)              # train set is all data but the validation
        validation_indices = list(validation_indices)
        # train the classifier on the training set #
        classifier.fit(data[train_indices], labels[train_indices])
        # use the trained classifier to make predictions #
        predictions = classifier.predict(data[validation_indices])
        # get the measure of the performance of the classifier #
        perf_measure = performance(labels[validation_indices], predictions, classes)

        # # append mcc for evaluation
        # scores[i] = mcc
        # print(scores)
        # store results

    ############################################################################################################
    # # CODE USED TO WRITE DATA TO TEXT FILES
        # write the results to file #
        # ofile.write(str(perf_measure) + '\n\n') # write the measures like true positive etc for each class
        # predictions_to_write = [str(j) for j in predictions]
        # validation_labels_to_write = [str(j) for j in validation_labels]
        # ofile.write('predictions:\n' + ','.join(predictions_to_write) + '\n\n')
        # ofile.write('actual labels:\n' + ','.join(validation_labels_to_write) + '\n\n')
        # confusion = confusion_matrix(labels[validation_indices], predictions)
        # mcc = matthews_corrcoef(labels[validation_indices], predictions)
    # ofile.write('\n' + '#' * 100 + '\n\n')
    # ofile.close()
    ############################################################################################################

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


def crossValidation(classifier,validationData, validationLabels, folds):
    return kfold(classifier, validationData, validationLabels, folds)


###################################################################################################################
"""
function to be removed in the final submission. Just to run and test the code
"""
###################################################################################################################
def run():
    data, val, datalabel, vallabel = ld.trainvalsplit('train')
    classifiers = []
    # for c in range(1,7):
    #     for deg in range(1,7):
    #         cf = SVC(C=c/10, kernel='poly', degree=deg, gamma=0.05)
    #         classifiers.append(cf)
    #
    # cf = SVC(C=0.5, kernel='poly', degree=1, gamma=0.05)
    dummy_size = 100
    td = data[:dummy_size]
    tl = datalabel[:dummy_size]

    cf1 = SVC(C=1, kernel='poly', degree=5, gamma=0.05)
    cf2 = KNeighborsClassifier(n_neighbors=5)
    cf3 = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,64), random_state=1)

    # kfold(cf1, td, tl, 10, 5)
    # kfold(cf2, td, tl, 10, 5)
    kfold(cf3, td, tl, 10, 5)
    # nnet_sizes = [[784, 100, 10], [784, 200, 10], [784, 300, 10], [784, 400, 10], [784, 500, 10],
    #               [784, 300, 200, 10], [784, 300, 100, 10], [784, 500, 200, 10], [784, 500, 100, 10],
    #               [784, 128, 64, 10]]
    #
    # for nnet_size in nnet_sizes:
    #     kfold([], nnet_size, data, datalabel, 5)


    # for i in range(1, 16):
    #     cf = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    #     classifiers.append(cf)
    #
    # for cf in classifiers:
    #     kfold(cf, [], data, datalabel, 10, 5)

    # cf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,64), random_state=1)
    # start = time.time()
    # cf.fit(data, datalabel)
    # print((time.time()-start))


if __name__ == '__main__':
    run()
