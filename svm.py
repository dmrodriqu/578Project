import LoadData as ld
import numpy as np
import crossval
from cuml.svm import SVC as cuSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def tuning(data, val, datalabel, vallabel):
    meanvar = []
    for i in range(1, 2):
        val.ravel()
        knearest = SVC(C = i)
        knearest.fit(data, datalabel)
        predictions = knearest.predict(val)
        print(predictions)
        print("performing cross-validation")
        #u, sig = crossval.kfold(knearest,val,vallabel, 5)
        print("crossval ended")
        print("mean = {:}, variance = {:}".format(u, sig))
        #meanvar.append([u,sig])
        #print(meanvar)
        print(confusion)
    return



data, validation, datalabel, validationlabel = ld.trainvalsplit('train')
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
dbin = lb.fit_transform(datalabel)
vbin = lb.fit_transform(validationlabel)
tuning(data[:1000], validation[:1000], datalabel[:1000], validationlabel[:1000])

