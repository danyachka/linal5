from sklearn.cluster import KMeans
from numpy import array
from sympy import Matrix
from colorama import Fore, Style
import numpy as np
import sympy as s

GET_ALL = -1


def getLowestEigenvectors(m: Matrix, k: int) -> [Matrix]:
    a = m.eigenvects()

    a = sorted(a, key=lambda x: x[0])

    vectors = []
    for val in a:
        for vect in val[2]:
            vectors.append(vect)

    if k >= len(vectors) or k == GET_ALL:
        return vectors

    result = []
    for i in range(k):
        result.append(vectors[i])

    return result


def getMatrixByList(vectors: [Matrix]) -> array:
    size = len(vectors)
    if size == 0:
        raise Exception("len = 0")

    a = []
    for x in vectors:
        a.append(list(x))
    res = np.array(a).transpose()
    return res


def getVectByHighestABSVal(m: Matrix) -> Matrix:
    values = m.eigenvects()

    val = None
    v = None
    for vect in values:
        tempVal = vect[0]

        if v is None:
            val = tempVal
            v = vect[2][0]
        elif abs(tempVal) > abs(val):
            val = tempVal
            v = vect[2][0]

    return v


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v


def convertSynpy2npArray(m: Matrix) -> array:
    return array(m.tolist())
