import LoadData as ld
import numpy as np
import crossval
   
def tuning(data, val, datalabel, vallabel):
    for i in range(1, 15):
        knearest = cuKNN(n_neighbors = i)
        knearest.fit(data,datalabel)
        predictions = knearest.predict(val)
        confusion = confusion_matrix(vallabel, predictions)
        print("performing cross-validation")
        u, sig = crossval.crossValidation(knearest,val,vallabel, 10)
        print("crossval ended")
        print("mean = {:}, variance = {:}".format(u, sig))
def tuning_nonGPU(data, val, datalabel, vallabel):
    for i in range(1, 15):
        knearest = neighbors.KNeighborsClassifier(n_neighbors = i, n_jobs = -1)
        print("performing cross-validation")
        u, sig = crossval.crossValidation(knearest,val,vallabel, 10)
        print("crossval ended")
        print("mean = {:}, variance = {:}".format(u, sig))


data, val, datalabel, vallabel = ld.trainvalsplit('train')
try:
    # use GPU 
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    from sklearn.metrics import confusion_matrix
    tuning(data, val, datalabel, vallabel)
except:
    # no GPU present
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    from sklearn.metrics import confusion_matrix
    tuning_nonGPU(data, val, datalabel, vallabel)
