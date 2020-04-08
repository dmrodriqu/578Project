import numpy as np
from math import erf
from scipy.stats import f as F
from scipy.stats import t as T
from itertools import combinations


# population mean
def mean(population):
    popsum = sum(population)
    return popsum/len(population)
# sum squares

def sumSquares(concatenatedPopulation, u):
    return sum([(sample - u)**2 for sample in concatenatedPopulation])

# sum of squares residual
def sumSquaresResid(population, groupMeans):
    ssresid = 0
    for i in range(len(population)):
        ssresid += sum((population[i] - groupMeans[i])**2)
    return ssresid

# total sum of squares
def totss(population, groupMeans, u):
    sse = 0
    ix = 0
    for i in population:
        n = len(i)
        means = np.repeat(groupMeans[ix],n)
        sse += sum((means - u)**2)
        ix += 1
    return sse

def createGroups(population):
    groupMeans = [np.apply_along_axis(mean, 0, s) for s in population]
    popcon = np.concatenate(population)
    u = popcon.mean()
    return popcon, groupMeans, u, len(groupMeans), len(popcon)

def calculatePValue(numbersamples, numbergroups, ssresid, ssexplained):
    df = numbersamples - numbergroups
    meansquaresres = ssresid/df
    dfex = numbergroups - 1
    meansquaresEx = ssexplained/dfex
    fScore = meansquaresEx/meansquaresres
    pval = 1- F.cdf(fScore, dfex, df)
    return meansquaresres, meansquaresEx, fScore, pval


# anova function to calculate 1 way anova
# take in population as array of group values
# outputs mean square residual, mean square of groups, group sum of squares
# f score, and p value
def anova(population):
    popcon, groupMeans, u, numbergroups, numbersamples = createGroups(population)
    ss = sumSquares(popcon, u)
    ssresid = sumSquaresResid(population, groupMeans)
    sse = totss(population, groupMeans , u)
    msres, msex, fscore, p = calculatePValue(numbersamples, numbergroups, ssresid, sse)
    return ss, ssresid, sse, msres, msex, fscore, p



A = [12.6, 12, 11.8, 11.9, 13, 12.5, 14]
B = [10, 10.2, 10, 12, 14, 13]
C = [0.1, 1, 1, 1.9, 8.9, 1.7, 1.6, 12]
D = [0.1, 12, 13, 1.9, .9, 1.7, 1.6, 12]



population =  [np.array(A), np.array(B), np.array(C), np.array(D)]
def ttest(s1, s2):
    # calculate means
    smu1 = s1.mean()
    smu2 = s2.mean()
    # calcuate standard deviation
    sigma1 = s1.std()
    sigma2 = s2.std()
    sigman1 = ((sigma1**2)/len(s1)) 
    sigman2 = ((sigma2**2)/len(s2))
    # get estimator
    ssq = (sigman1 + sigman2)
    s = ssq**(1/2)
    #calculate degrees of freedom unequal variance
    df  = (ssq**2)/(((sigman1**2)/(len(s1)-1)) + (sigman2**2)/(len(s2)-1))
    # calculate t
    t = (smu1 - smu2)/s
    # difference != 0
    oneTail = (T.cdf(abs(t), df=df))
    alpha = 0.05
    if (2*(1-oneTail)) < alpha:
        # difference is > 0
        if (1-oneTail) < alpha:
            print(1-oneTail)
            return 1
        # difference is < 0
        else:
            print(oneTail)
            return 2
    return 0
print(anova(population))

def postHoc(population):
    winners = []
    losers = []
    for i,j in combinations(range(len(population)), 2):
        if (ttest(population[i],population[j])) == 1:
            winners.append(population[i])
        else:
            losers.append(population[j])
    if not bool(winners):
        return postHoc(winners)
    else:
        return winners, losers
print(postHoc(population)[0])

        

