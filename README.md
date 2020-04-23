Simply download all the code files provided into a folder, let us call it project_folder. Then change directory to the project_folder. Please make sure that within the project_folder, there is another folder titled data that has all the MNIST  dataset. 

--------------------------------------------------
Running the program
--------------------------------------------------
Then code is run by the entering a command in the following format in command prompt (e.g. terminal): 
python main.py --option=t --data_percentage=p --output_file=<filepath>

where 
- t is 1 for SVM, 2 for NN, and 3 for KNN. It is set to 1 by default.

- data_percentage is a number from 0 to 100 signifying what percentage of training data should be used during the hyperparameter tuning stage. It is set to 100 by default. Note after the hyperparameter tuning stage, full training dataset is used to train the model with the best hyperparamters found.

- <filepath> is the full path to an output file to which the results during hyperparameter tuning shall be writted to.

E.g the following 
python main.py --option=1 --data_percentage=50 --output_file=./output.txt 

Note. Because some files are generated during the course of running the program, you might need to use write permissions. In Unix/Linux, this is done by having "sudo" before the command above. E.g.

python main.py --option=1 --data_percentage=50 --output_file=./output.txt 

--------------------------------------------------
Various outputs for the program
--------------------------------------------------
This shall load the training and testing datasets from the project_folder/data folder. It shall begin the 5-fold cross-validation on the classifier of choice using the models with the hyperparameters mentioned in section 3.1 of the report. Once the cross validation finishes, the model with the best hyperparameters is chosen from the set of all models, and this best model is then trained on the full training data set. After training, this model is used to predict the labels for each of the samples in the test dataset. The MCC score on the test dataset, along with accuray, and per class precision and recall are displayed on the console. 

During each run of the 5-folds cross-validation for a model settings, the following information is written to the output.txt file: 
1. Model specifications, e.g. C: 0.6, kernel: poly, degree: 1, gamma: 0.05 for SVM

2. fold number, e.g. fold #: 1

3.a dictionary which contains the true positives (TP), false positives, true negatives (TN), false      negatives (FN), and the number of samples in the validation dataset for each class (count). This is computed after the current model trains on the current training set and makes predictions on the validation dataset. This is a Python dictionary. A snippet from such a dictionary is {0: {'TP': 1179, 'FP': 44, 'TN': 10757, 'FN': 20, 'count': 1199}, 1: {'TP': 1314, 'FP': 47, 'TN': 10609, 'FN': 30, 'count': 1344}, ... 9: {'TP': 1156, 'FP': 86, 'TN': 10665, 'FN': 93, 'count': 1249}}

4. predictions: these are the predicted labels for each of the samples in the validation dataset. E.g. 1,0,4,7,..

5. actual: these are the ground truth labels for each of the samples in the validation dataset. E.g. 1,2,4,7,..

The information described above is written to the output.txt file for each of the model, for all the five runs of the cross validation algorithm, hence this file will have a lot of material in it after the program finishes the cross validation. Please note that the information is appended to output.txt, so if it contains any data previously, that would not be lost.

Finally, after tuning the hyperparameters, when the best model is tested on the test dataset after being trained on the full training dataset, an additional file "final_output.txt" is generated. This file has the following information: 

1. predictions: same as predictions in the output.txt file, depicting the model's predictions on the test dataset

2. actual: same as actual in the output.txt file, for the test dataset.

--------------------------------------------------
Visualization
--------------------------------------------------
The file visualization.py on line 26 has three hardcoded file names. These are the files that contain the information written to during performing the 5-fold cross-validation. This is the same file selected initially at running the code, e.g. using 

python main.py --option=1 --output_file=./results-SVM.txt 

To view the box plots etc., (see section Results in the report) kindly have these three files 'results-KNN.txt', 'results-SVM.txt', 'results-NN.txt' present in the folder containing all the code files, and simply run the validation.py file using

python visualization.py

Please note that the code appends to files during parameter tuning via cross validation. So for each selection of the classifier (from SVM, KNN, Nueral networks), use the appropriate name for the output file (or rename before running visualization.py)

There are sample files in the source folder which can be used to quickly view the results.

--------------------------------------------------
Running time
--------------------------------------------------
As the full training dataset contains 60,000 samples, running the classifiers, especially SVM, can take a long time (> 6 hours on our machine). The training dataset to be used during hyperparameter tuning phase can be adjusted using the --data_percentage argument when executing the program.

Additionally, you may choose to run crossfold validation in parallel with the argument
--n_jobs [workers]

-1 worker for all threads, -2 for all but one.
n workers for n threads
