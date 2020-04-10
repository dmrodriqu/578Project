import LoadData as ld
import numpy as np
import crossval
import matplotlib.pyplot  as plt

def tuning(data, val, datalabel, vallabel):
    meanvar = []
    for i in range(1, 15):
        knearest = cuKNN(n_neighbors = i)
        knearest.fit(data,datalabel)
        predictions = knearest.predict(val)
        confusion = confusion_matrix(vallabel, predictions)
        print("performing cross-validation")
        u, sig = crossval.kfold(knearest,val,vallabel, 5)
        print("crossval ended")
        print("mean = {:}, variance = {:}".format(u, sig))
        meanvar.append([u,sig])
    return meanvar
def tuning_nonGPU(data, val, datalabel, vallabel):
    meanvar = []
    for i in range(1, 15):
        knearest = neighbors.KNeighborsClassifier(n_neighbors = i, n_jobs = -1)
        print("performing cross-validation")
        u, sig = crossval.crossValidation(knearest,val,vallabel, 5)
        print("crossval ended")
        print("mean = {:}, variance = {:}".format(u, sig))
        meanvar.append([u,sig])
    return meanvar

data, val, datalabel, vallabel = ld.trainvalsplit('train')
# use GPU 
try:
    print('gpu')
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    from sklearn.metrics import confusion_matrix
    print('showing')
    plt.imshow(data[20].reshape(28,28))
    plt.show()
    print(tuning(data, val, datalabel, vallabel))

except:
    print("cpu")
    # no GPU present
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    from sklearn.metrics import confusion_matrix
    print('showing')
    plt.imshow(data[20].reshape(28,28))
    plt.show()
    tuning_nonGPU(data, val, datalabel, vallabel)
