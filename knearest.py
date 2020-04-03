import LoadData as ld
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from cuml.neighbors import KNeighborsClassifier as cuKNN
   
data, val, datalabel, vallabel = ld.trainvalsplit('train')

k = 10
knearest = cuKNN(n_neighbors = k)
knearest.fit(data,datalabel)
predictions = knearest.predict(val)






