import numpy as np


def get_transitions(X, A, p, q_low, q_high):
    P = np.zeros((X, X, A))
    for i in range(0, X):
        P[i, i, 0] = (1 - p) * (1 - q_low) + p * q_low
        P[i, i, 1] = (1 - p) * (1 - q_high) + p * q_high
        if i + 1 < X:
            P[i, i + 1, 0] = p * (1 - q_low)  # 0 corresponds to low action
            P[i, i + 1, 1] = p * (1 - q_high)  # 1 corresponds to high action
        if i - 1 > 0:
            P[i, i - 1, 0] = q_low * (1 - p)  # 0 corresponds to low action
            P[i, i - 1, 1] = q_high * (1 - p)  # 1 corresponds to high action
            #managing edge cases
        if i == 0:
            P[i, i, 0]=P[i, i, 0]+q_low * (1 - p)  # 0 corresponds to low action
            P[i, i, 1] = P[i, i, 1] + q_high * (1 - p)  # 0 corresponds to low action
        if i == X-1:
            P[i, i, 0] = P[i, i, 0] + p * (1 - q_low)  # 0 corresponds to low action
            P[i, i, 1] = P[i, i, 1] + p * (1 - q_high)  # 0 corresponds to low action

        #random truck overload

        #P[i,min(i+50,X-1),0]=0.1
        #P[i, min(i + 50, X - 1), 1] = 0.2
    return P
