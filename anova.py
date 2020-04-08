import numpy as np
from scipy.stats import f as F

# population mean
def mean(population):
    popsum = sum(population)
    return popsum/len(population)
# sum squares

def sumSquares(concatenatedPopulation):
    return sum([(sample - u)**2 for sample in concatenatedPopulation])

# sum of squares residual
def sumSquaresResid(population, groupMeans):
    ssresid = 0
    for i in range(len(population)):
        ssresid += sum((population[i] - groupMeans[i])**2)
    return ssresid

# total sum of squares
def totss(population, groupMeans):
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
    meansquaresEx = sse/dfex
    fScore = meansquaresEx/meansquaresres
    pval = 1- F.cdf(fScore, dfex, df)
    return meansquaresres, meansquaresEx, fScore, pval


# anova function to calculate 1 way anova
# take in population as array of group values
# outputs mean square residual, mean square of groups, group sum of squares
# f score, and p value
def anova(population):
    popcon, groupMeans, u, numbergroups, numbersamples = createGroups(population)
    ss = sumSquares(popcon)
    ssresid = sumSquaresResid(population, groupMeans)
    sse = totss(population, groupMeans)
    msres, msex, fscore, p = calculatePValue(numbersamples, numbergroups, ssresid, sse)
    return ss, ssresid, sse, mres, msex, fscore, p
