import numpy as np
from utils.params import *


def fine_feature(X):
    F = np.eye((X))
    return F


def coarse_feature(X):
    F = np.zeros((int(X / 5), X))
    for k in range(0, F.shape[0]):
        i = k + 1
        j = 5 * (i - 1)
        while j <= 5 * i - 1:
            if 0 <= j < X:
                F[k, j] = 1
            j = j + 1
    return F


def pl_feature(X):  # piece-wise linear feature map
    F = np.zeros((2 * int(X / 5), X))
    F_coarse = coarse_feature(X)
    F[:int(X / 5), :] = F_coarse
    for k in range(0, int(F.shape[0]/2)):

        i = k+1
        j = 5 * (i - 1)
        while j <= 5 * i - 1:
            if 0 <= j < X:
                F[int(X/5)+k, j] = (j - 5 * (i - 1)) / 5
            j = j + 1

    return F
