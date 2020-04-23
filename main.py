from anova import *
from knearest import *
from nnet import *
from crossval import *
from svm import *
from visualization import *
import argparse
import LoadData as ld
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
import metric


def post_tuning(train_data, train_labels, test_data, num_classes, classifier):
    """
    takes the best classifier after tuning hyperparameters, and trains and tests it on the full training dataset
    and then makes predictions on the test dataset
    :param train_data: the full training dataset (includes the validation dataset as well)
    :param train_labels: the full training dataset labels
    :param test_data: the test dataset
    :param num_classes: number of classes in the dataset (10)
    :param classifier: a sklearn classifier
    :return: None
    """
    print('Training the classifier above on full training dataset')
    # train this best classifier on the full data #
    classifier.fit(train_data, train_labels)
    print('Training complete. Now we shall evaluate this trained classifier on the test dataset')
    predictions = classifier.predict(test_data['data'])
    # write predictions and actual labels to a file #
    ofile = open('final_output.txt', 'w+')
    ofile.write('This file has the actual labels and predicted labels for the test set\n\n')
    ofile.write('predictions:\n' + ','.join([str(i) for i in predictions]) + '\n\n')
    ofile.write('actual labels:\n' + ','.join([str(i) for i in test_data['label']]) + '\n\n')
    ofile.close()
    C = metric.confusionMatrix(test_data['label'], predictions)
    mcc = metric.matthews(C)
    accuracy = metric.accuracy(test_data['label'], predictions)
    precision, recall = metric.precisionrecall(C)
    print("The classifier's performance measures on the test set are:\n ")
    print("Mathew's correlation coefficient (MCC):", mcc)
    print("Accuracy:", accuracy)
    print("-" * 33)
    print("{:^2}{:^18}{:^15}".format("Class", "Precision", "Recall"))
    print("-"*33)
    for i in range(num_classes):
        print("{:^2}{:^22,.2f}{:>9,.2f}".format(i, precision[i], recall[i]))
    matrixOfConfusion = confusionMatrix(test_data['label'], predictions, normalized = True)
    fig, ax = plt.subplots()
    pos = ax.imshow(matrixOfConfusion, cmap = 'Blues', interpolation = None)
    ax.set_title("Confusion Matrix of Current Algorithm")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predictions")
    fig.colorbar(pos, ax=ax)
    plt.show()


def tune_hyperparams(classifiers, train_set, train_labels, num_classes, k, output_file, njobs = 1):
    """
    calls the kfold() function on all the classifiers, followed by ANOVA and post-hoc Tukey's HSD tests to
    determine the hyperparameters values for which we get the best performance on validation dataset
    :param classifiers: a list of sklearn classifiers. Type: Python list
    :param train_set: training dataset. Type: numpy matrix
    :param train_labels: training labels. Type: numpy array
    :param num_classes: 10. Type: int
    :param k: k-fold cross validation. Type: int
    :param output_file: path to file to write the cross validation data to
    :return: the index of the classifier in the list classifiers with the best performance
    """
     
    '''
    mcc_measures = np.zeros((len(classifiers), k))
    for i in range(len(classifiers)):
        mcc_measures[i, :] = kfold(classifiers[i], train_set, train_labels,
                                   num_classes, k, output_file)
    best_ind = statistical_analysis(mcc_measures)  # index of the best classifier

    '''

    #run in parallel
    mcc_measures = np.zeros((len(classifiers), k))
    print('running in parallel, order of folds may not display correctly')
    measures = Parallel(n_jobs = njobs, verbose = 1)(delayed(kfold)(classifiers[i], train_set, train_labels, num_classes, k, output_file) for i in range(len(classifiers)))
    for i in range(len(measures)):
        mcc_measures[i,:] = measures[i]

    return statistical_analysis(mcc_measures)


def run():
    """
    routine to run the full program
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=1,
                        help='Option to run (1 = SVM, 2 = Neural Network, 3 = K-Nearest Neighbors')
    parser.add_argument('--data_percentage', type=float, default=100,
                        help='percentage of training data to be used (a number in the interval (0-100]')
    parser.add_argument('--output_file', type=str, default='results.txt', help='Output file')
    parser.add_argument('--n_jobs', type=str, default=1, help='number of jobs')

    args = parser.parse_args()

    data, val_data, data_labels, val_labels = ld.trainvalsplit('train')  # train and validation datasets
    train_data = np.concatenate((data, val_data), axis=0)
    train_labels = np.concatenate((data_labels, val_labels), axis=0)
    test_data = ld.trainvalsplit('test')   # test set
    num_classes = 10
    k = 5   # number of folds
    ##################################################################
    # check the arguments entered #
    if args.data_percentage <= 0 or args.data_percentage > 100:
        print("kindly select a percentage number in (0, 100]")
        return
    if args.option not in [1, 2, 3]:
        print('Invalid option selected. Kindly use --help to see instructions as to how to run the program')
        return
    if not args.output_file:
        print('please enter a valid output file')
        return
    data_size = int((args.data_percentage/100)*train_data.shape[0])
    selected_trainset = train_data[0:data_size, :]
    selected_trainlabels = train_labels[0:data_size]
    ##################################################################

    if args.option == 1:    # run SVM
        print('SVM classifiers chosen. Tuning hyperparameters. \n'
              'All results shall be appended to the output file.')
        classifiers = []
        for c in range(1,7):
            for deg in range(1,7):
                cf = SVC(C=c/10, kernel='poly', degree=deg, gamma=0.05)
                classifiers.append(cf)

        best_ind = tune_hyperparams(classifiers, selected_trainset, selected_trainlabels,
                                    num_classes, k, args.output_file, int(args.n_jobs))
        print('based on statistical analysis of the 5-fold cross validation, the best SVM has C =',
              classifiers[best_ind].C, "and the degree of its kernel polynomial =", classifiers[best_ind].degree)
        post_tuning(selected_trainset, selected_trainlabels, test_data, num_classes, classifiers[best_ind])

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

        best_ind = tune_hyperparams(classifiers, selected_trainset, selected_trainlabels,
                                    num_classes, k, args.output_file)
        print('based on statistical analysis of the 5-fold cross validation, the best Neural Network has '
             "configuration " + '784,' + ','.join([str(i) for i in tuple(classifiers[best_ind].hidden_layer_sizes)])
              + ',10')
        post_tuning(selected_trainset, selected_trainlabels, test_data, num_classes, classifiers[best_ind])

    elif args.option == 3:  # run KNN classifiers
        print('KNN classifiers chosen. Tuning hyperparameters. \n'
              'All results shall be appended to the output file.')
        classifiers = []
        for i in range(1, 16):
            cf = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
            classifiers.append(cf)

        best_ind = tune_hyperparams(classifiers, selected_trainset, selected_trainlabels,
                                    num_classes, k, args.output_file)
        print('based on statistical analysis of the 5-fold cross validation, the best KNN classifier is the one with',
              classifiers[best_ind].n_neighbors, "neighbors")
        post_tuning(selected_trainset, selected_trainlabels, test_data, num_classes, classifiers[best_ind])

run()
