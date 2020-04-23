from anova import *
from knearest import *
from nnet import *
from crossval import *
from metric import *
from svm import *
from visualization import *
import argparse
import LoadData as ld
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from math import ceil
from sklearn.svm import SVC, LinearSVC
import nnet
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
import matplotlib.pyplot as plt


def post_tuning(train_data, train_labels, test_data, num_classes, classifiers, mcc_matrix):
    best_ind = statistical_analysis(mcc_matrix)  # index of the best classifier
    print('based on statistical analysis of the 5-fold cross validation, the best classifier is',
          classifiers[best_ind])
    print('training the classifier above on full training dataset')
    # train this best classifier on the full data #
    classifiers[best_ind].fit(train_data, train_labels)
    print('training done. Now we shall evaluate this trained classifier on the test dataset')
    predictions = classifiers[best_ind].predict(test_data['data'])
    # write predictions and actual labels to a file #
    ofile = open('final_output.txt', 'w+')
    ofile.write('This file has the actual labels and predicted labels for the test set\n\n')
    ofile.write('predictions:\n' + ','.join([str(i) for i in predictions]) + '\n\n')
    ofile.write('actual labels:\n' + ','.join([str(i) for i in test_data['label']]) + '\n\n')
    ofile.close()
    matrixOfConfusion = confusionMatrix(test_data['label'], predictions, normalized = True)
    fig, ax = plt.subplots()
    pos = ax.imshow(matrixOfConfusion, cmap = 'Blues', interpolation = None)
    ax.set_title("Confusion Matrix of Current Algorithm")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predictions")
    fig.colorbar(pos, ax=ax)
    plt.show()
    mcc = compute_MCC(test_data['label'], predictions, list(range(num_classes)))
    print("The Mathew's correlation coefficient for this classifier on the test dataset is: ", mcc)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=1,
                        help='Option to run (1 = SVM, 2 = Neural Network, 3 = K-Nearest Neighbors')
    parser.add_argument('--output_file', type=str, default='results.txt', help='Output file')
    parser.add_argument('--njobs', type=str, default='1', help='number of jobs (if available)')
    args = parser.parse_args()

    data, val_data, data_labels, val_labels = ld.trainvalsplit('train')  # train and validation datasets
    train_data = np.concatenate((data, val_data), axis=0)
    train_labels = np.concatenate((data_labels, val_labels), axis=0)




    test_data = ld.trainvalsplit('test')   # test set
    num_classes = 10
    k = 5   # number of folds
    #################################################################
    tsize = 100     # just
    ##################################################################
    if args.option == 1:    # run SVM
        print('SVM classifiers chosen. Tuning hyperparameters. \n'
              'All results shall be appended to the output file.')
        classifiers = []
        for c in range(1,7):
            for deg in range(1,7):
                cf = SVC(C=c/10, kernel='poly', degree=deg, gamma=0.05)
                classifiers.append(cf)

            '''
            # emb par loop
            results = Parallel(n_jobs=4)(delayed(SVC)(C=c/10, kernel='poly', degree=deg, gamma=0.05) for deg in range(1,7))
            for i in results:
                classifiers.append(i)
        for i in range(len(classifiers)):
            mcc_measures[i, :] = kfold(classifiers[i], train_data, train_labels, num_classes, k, args.output_file)
        '''  

        #run in parallel
        mcc_measures = np.zeros((len(classifiers), k))

        print('running in parallel, order of folds may not display correctly')
        measures = Parallel(n_jobs = int(args.njobs), verbose = 100)(delayed(kfold)(classifiers[i], train_data, train_labels, num_classes, k, args.output_file) for i in range(len(classifiers)))
        for i in range(len(measures)):
            mcc_measures[i,:] = measures[i]

        post_tuning(train_data, train_labels, test_data, num_classes, classifiers, mcc_measures)

    elif args.option == 2:  # run Neural Network classifiers
        print('Neural Network classifiers chosen. Tuning hyperparameters. \n'
              'All results shall be appended to the output file.')
        classifiers = []
        nnet_sizes = [[100], [200], [300], [400], [500],
                      [300, 200], [300, 100], [500, 200], [500, 100], [128, 64]]
        for net_size in nnet_sizes:
            cf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,
                                              hidden_layer_sizes=tuple(net_size), random_state=1)
            classifiers.append(cf)
        mcc_measures = np.zeros((len(classifiers), k))
        for i in range(len(classifiers)):
            mcc_measures[i, :] = kfold(classifiers[i], train_data, train_labels, num_classes, k, args.output_file)
        post_tuning(train_data, train_labels, test_data, num_classes, classifiers, mcc_measures)

    elif args.option == 3:  # run KNN classifiers
        print('KNN classifiers chosen. Tuning hyperparameters. \n'
              'All results shall be appended to the output file.')
        classifiers = []
        for i in range(1, 16):
            cf = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
            classifiers.append(cf)

        mcc_measures = np.zeros((len(classifiers), k))
        for i in range(len(classifiers)):
            mcc_measures[i,:] = kfold(classifiers[i], train_data, train_labels, num_classes, k, args.output_file)
        post_tuning(train_data, train_labels, test_data, num_classes, classifiers, mcc_measures)
    else:
        print('kindly use --help to see instructions as to how to run the program')
run()
