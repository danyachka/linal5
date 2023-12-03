from sklearn.cluster import KMeans
from sympy import Matrix
from numpy import array
import numpy as np

import utils
import reader


def firstPart():
    k = 7
    m: Matrix = reader.getFromTable("linal5matrix.csv")

    v: array = utils.getMatrixByList(utils.getLowestEigenvectors(m, k))

    km = KMeans(n_clusters=k, n_init="auto").fit(v)
    print("k = " + str(k))
    print(km.labels_)


def secondTask():
    m: Matrix = reader.getFromTable("linal5m2.csv")

    v = utils.getVectByHighestABSVal(m)
    print(v)

    res = utils.pagerank(utils.convertSynpy2npArray(m), 100, 0.85)
    print(res)


if __name__ == '__main__':
    #firstPart()
    secondTask()
