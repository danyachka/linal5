from sklearn.cluster import KMeans
from sympy import Matrix
from numpy import array
import numpy as np
from colorama import Fore, Style

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
    print(Fore.BLUE + "Собственный вектор с наибольшим по модулю собственным числом - "
          + Fore.GREEN + str(v.tolist()) + Fore.RESET)

    res = utils.pagerank(utils.convertSynpy2npArray(m), 100, 0.85)
    numberedRes = []
    size = len(res)
    for i in range(size):
        number = res[i]
        numberedRes.append((i + 1, number))

    numberedRes = sorted(numberedRes, key=lambda x: x[1])
    numberedRes.reverse()

    print(Fore.GREEN + "\nТаблица рейтинга" + Fore.RESET)
    for i in range(size):
        r = numberedRes[i]
        print(Fore.BLUE + str(i + 1) + ")" + Fore.RESET + " Номер - " + str(r[0]) + " | Рейтинг - " + str(r[1]))


if __name__ == '__main__':
    #firstPart()
    secondTask()
