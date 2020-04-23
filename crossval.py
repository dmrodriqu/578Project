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
import metric


def createix(folds):
    foldarr = []
    for i in range(folds):
        # create folds
        ix = [x%folds for x in range(i, i+folds)]
        loo = ix[0]
        fol = np.array(ix[1:])
        foldarr.append([loo,fol])
    return np.array(foldarr)


def write_to_file(classifier, perf_measure, fold_num,
                  actual_labels, predictions, ofilename,
                  write_classifier=False):
    """
    writes the performance of the classifier to a text file
    :param classifier: the sklearn classifier. Type: one of skleran SVC, Multilayer Perceptron, or K-Nearest Neighbors
    :param perf_measure: the data of true positives, false positives, true negatives, and false negatives. Type: dictionary
    :param fold_num: number of the current fold. Type: int
    :param actual_labels: the ground truth labels for the data set. Type: numpy array
    :param predictions: the predicted labels for the data set. Type: numpy array
    :param ofilename: output file name. Type: String
    :return: None
    """
    cf_svm = SVC(C=1, kernel='poly', degree=5, gamma=0.05)
    cf_knn = KNeighborsClassifier(n_neighbors=5)
    cf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,64), random_state=1)
    ofile = open(ofilename, 'a+')
    if write_classifier:
        if type(classifier) == type(cf_svm):    # the classifier is an SVM
            ofile.write("C: " + str(classifier.C) +
                        ", kernel: " + classifier.kernel + ", degree: " + str(classifier.degree)
                        + ", gamma: " + str(classifier.gamma) +"\n\n")

        elif type(classifier) == type(cf_nn):   # the classifier is a neural network
            ofile.write("NNET Size: " + '784,' +
                        ','.join([str(i) for i in tuple(classifier.hidden_layer_sizes)]) + ',10' +'\n\n')

        elif type(classifier) == type(cf_knn):  # the classifier is a KNN
            ofile.write('K-Nearest Neighbors. # of Clusters: ' + str(classifier.n_neighbors) + '\n\n')

    ofile.write("fold #: " + str(fold_num) + '\n\n')
    ofile.write(str(perf_measure) + '\n\n')
    ofile.write('predictions:\n' + ','.join([str(i) for i in predictions]) + '\n\n')
    ofile.write('actual labels:\n' + ','.join([str(i) for i in actual_labels]) + '\n\n')
    ofile.close()


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


def confusion_mat(actual_labels, predicted_labels, classes):
    """
    computes the multiclass confusion matrix
    :param actual_labels: actual labels for each data instance. Type: numpy array
    :param predicted_labels: predicted labels for each data instance. Type: numpy array
    :param classes: a list of all classes. Type: list
    :return: confusion matrix. The rows are the predictions, and the columns actual labels. Type: numpy matrix
    """
    try:
        assert actual_labels.shape == predicted_labels.shape
    except:
        print('the sizes of actual labels and predictions do not match')
        return
    mat = np.zeros((len(classes), len(classes)))
    for k in classes:
        for i in range(actual_labels.shape[0]):
            if actual_labels[i] == k:
                mat[k][predicted_labels[i]] += 1
    return mat


def compute_MCC(actual_labels, predictions, classes):
    """
    function compute_MCC(actual_labels, predictions, classes)
    computes the Mathew's Correlation coefficient for a classifier's performance
    :param actual_labels: actual labels for each data instance. Type: numpy array
    :param predictions: predicted labels for each data instance. Type: numpy array
    :param classes: a list of all classes. Type: list
    :return: the computed Mathew's correlation coefficient. Type: float
    """
    C = metric.confusionMatrix(actual_labels, predictions)
    # C = confusion_mat(actual_labels, predictions, classes)
    # use the confusion matrix to compute the multiclass Mathew's correlation coefficient #
    t = np.sum(C, axis=1)  # t[j] := number of times class j truly occurred
    p = np.sum(C, axis=0)  # p[j] := number of times class j was predicted
    c = np.trace(C)  # c := total number of correct predictions
    s = actual_labels.shape[0]  # s:= total number of data points
    return ((c * s) - np.dot(t, p)) / np.sqrt((s ** 2 - np.sum(np.square(p))) * (s ** 2 - np.sum(np.square(t))))


def kfold(classifier, data, labels, num_classes, k, ofile):
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
    :param ofile: the output file results need to be written to
    :return: Mathew's correlation coefficient computed for each fold. Type: numpy array
    """
    # split the data into folds many disjoint sets #
    n = data.shape[0]    # the total number of data points
    subset_size = ceil(n / k)
    all_indices = set(range(n))
    scores = np.array(np.zeros(k))
    classes = list(range(num_classes))
    cf_svm = SVC()
    cf_knn = KNeighborsClassifier()
    cf_nn = neural_network.MLPClassifier()
    if type(classifier) == type(cf_svm):
        print("SVM. C =", classifier.C, "kernel using polynomial of degree", classifier.degree)
    elif type(classifier) == type(cf_knn):
        print("KNN classifier with neighbors =", classifier.n_neighbors)
    elif type(classifier) == type(cf_nn):
        print("NNET Size: " + '784,' + ','.join([str(i) for i in tuple(classifier.hidden_layer_sizes)]) + ',10')
    for i in range(k):
        print("fold #: ", i)
        validation_indices = set(range(subset_size*i,min(subset_size*(i+1), n)))  # the indices of the validation set
        train_indices = list(all_indices - validation_indices)              # train set is all data but the validation
        validation_indices = list(validation_indices)
        actual = labels[validation_indices]
        # train the classifier on the training set #
        classifier.fit(data[train_indices], labels[train_indices])
        # use the trained classifier to make predictions #
        predictions = classifier.predict(data[validation_indices])
        # get the measure of the performance of the classifier #
        perf_measure = performance(actual, predictions, classes)
        mcc = compute_MCC(labels[validation_indices], predictions, classes)
        scores[i] = mcc
        write_to_file(classifier, perf_measure, i, labels[validation_indices], predictions, ofile,
                      bool(i == 0))
        if i == 4:
            output_file_handler = open(ofile, "a+")
            output_file_handler.write("\n" + '#'*100 + "\n\n")
            output_file_handler.close()
    return scores


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

    # kfold(cf2, td, tl, 10, 5)
    # kfold(cf1, td, tl, 10, 5)
    # kfold(cf2, td, tl, 10, 5)
    # kfold(cf3, td, tl, 10, 5)
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
    # act = np.array([0, 1, 1, 2])
    # preds = np.array([0, 1, 2, 1])
    actual = np.random.randint(0,10,10000)
    predictions = np.random.randint(0,10,10000)
#     C = confusion_mat(act, preds, list(range(10)))
#     # print(C)
#     t = np.sum(C, axis=1)
#     p = np.sum(C, axis = 0)
#     c = np.trace(C)
#     s = act.shape[0]
#
#     numerator = (c*s) - np.dot(t, p)
#     den = np.sqrt((s**2 - np.sum(np.square(p))) * (s**2 - np.sum(np.square(t))))
#     mcc = numerator/den
#     print(mcc, matthews_corrcoef(act, preds))
#
    # scores, predictions, actual = kfold(cf2, td, tl, 10, 5, '')
    # predictions = np.array([7, 0, 7, 0, 7, 7, 1, 1, 1, 1, 9, 0, 8, 7, 0, 1, 3, 3, 9, 3])
    # actual = np.array([7, 5, 5, 0, 7, 7, 8, 1, 1, 1, 9, 0, 8, 7, 0, 1, 5, 5, 9, 3])
    C = confusion_mat(actual, predictions, list(range(10)))
    C1 = metric.confusionMatrix(actual, predictions)
    print(metric.matthews(C1))
    print(matthews_corrcoef(actual,predictions))
    t = np.sum(C, axis=1)
    p = np.sum(C, axis = 0)
    c = np.trace(C)
    s = actual.shape[0]

    numerator = (c*s) - np.dot(t, p)
    den = np.sqrt((s**2 - np.sum(np.square(p))) * (s**2 - np.sum(np.square(t))))
    mcc = numerator/den
    print(mcc)
    print(metric.matthews(C))
#     print(matthews_corrcoef(actual, predictions))
#     print(compute_MCC(actual, predictions, list(range(10))))
# if __name__ == '__main__':
#     run()
