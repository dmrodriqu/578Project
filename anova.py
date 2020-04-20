import numpy as np
from math import erf, sqrt
from scipy.stats import f as F
from scipy.stats import t as T
from itertools import combinations
from statsmodels.stats.libqsturng import psturng, qsturng

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


def tukeyhsd(results, msres):
    """
    function tukeyhsd(results)
    runs the Tukey HSD test on the results to find which of the population mean is statistically significantly
    different from the others
    :param results: a matrix where each row represents the data for one population. Type: list of (list of numbers)
    :return: a list of lists, where each small list has two integers, i and j, which indicate that classifier i
             and classifier j have means which are statistically significantly different
    """

    n, d = results.shape
    # 1. Compute the absolute value of the difference of means of each pair of classes
    diff_of_means = {}
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            diff_of_means[(i,j)] = (abs(results[i].mean() - results[j].mean()))
    # 2. Compute the degrees of freedom
    df = n*d - n
    # 3. Fix an alpha
    alpha = 0.05
    # 4. Find out the q value from the Q table for Tukey's test
    q = qsturng(1-alpha, n, df)
    # 5. Compute critical range
    qu = q*sqrt(msres/d)    # critical range = q*sqrt(MS_w/2 * (1/d + 1/d)) = q*sqrt(MS_w / d)
    # 6. Find the pairs of classes whose absolute difference of means exceed the critical range
    significantly_different_pairs = []
    for key in diff_of_means:
        if diff_of_means[key] > qu:
            significantly_different_pairs.append(key)

    return significantly_different_pairs


def statistical_analysis(population):
    """
    performs the one-way ANOVA on the population first to see if the null hypothesis, i.e. the means of each sample
    is the same, is rejected. If so, then the function performs the Tukey's HSD test to find out the pairs of samples
    whose means are statistically significantly different from each other. The alpha value we select is 0.05 (5%)

    :param population: a matrix whose rows are the individual samples. Type: numpy matrix
    :return: the index of the best sample. Type: int
    """
    ss, ssresid, sse, msres, msex, fscore, p = anova(population)
    if p > 0.05:    # all samples had means which were statistically similar
        return 0    # we can select one sample randomly. Here we just say select the 0-th (first) sample
    # if p < 0.05, we reject the null-hypothesis that all means were equal. Now we need to find the best sample,
    # i.e the one with the highest mean
    different_pairs = tukeyhsd(population, msres)
    # determine the best sample #
    sample_means = {}
    for i in range(population.shape[0]):
        sample_means[i] = population[i].mean()

    best_mean = -10  # starting with a dummy value for the best_mean. Selected -10 because MCC will never be < -1
    best_sample_num = -1    # starting with a dummy value for best sample
    for pair in different_pairs:
        if best_mean < sample_means[pair[0]]:
            best_mean = sample_means[pair[0]]
            best_sample_num = pair[0]
        elif best_mean < sample_means[pair[1]]:
            best_mean = sample_means[pair[1]]
            best_sample_num = pair[1]

    return best_sample_num



# results = np.array([[7, 8, 15, 11, 9, 10],
#            [12, 17, 13, 18, 19, 15],
#            [14, 18, 19, 17, 16, 18],
#            [19, 25, 22, 23, 18, 20]])
#
# res2 = np.array([[7, 7, 15, 11, 9],
#                 [12, 17, 12, 18, 18],
#                  [14, 18, 18, 19, 19],
#                  [19, 25, 22, 19, 23],
#                  [7, 10, 11, 15, 11]])
#
# print(statistical_analysis(results))
# print(statistical_analysis(res2))