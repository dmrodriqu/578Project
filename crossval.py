import LoadData as ld
import numpy as np

print("loading data")
data, val, datalabel, vallabel = ld.trainvalsplit('train')

folds = 5
ix = np.arange(len(data))
splits = np.array_split(ix, folds)

def createix(folds):
    foldarr = []
    for i in range(folds):
        # create folds
        ix = [x%folds for x in range(i, i+folds)]
        loo = ix[0]
        fol = np.array(ix[1:])
        foldarr.append([loo,fol])
    return np.array(foldarr)

foldix = createix(folds)

# for every fold
for i in range(len(foldix)):
    # get the split ix
    foldindices = foldix[i][1]
    # get the indices of every datum in split
    crossvalix = np.array(splits)[foldindices]
    validation = data[np.concatenate(crossvalix)]
    leaveoneout = data[np.array(np.array(splits)[foldix[i][0]])]
    # call classifier 
    # store results
