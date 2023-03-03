from utils.feature_space import *
import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.params import *
from utils.policy_eval import *
from utils.approx_policy_iter import *
from utils.line_plot import *


def td0_eval():
    # lazy policy
    policy = pi_lazy(X, A)
    a = 100000
    b = 100000
    F1 = fine_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters", 'Fine feature map (lazy)')
    plt1.show()

    policy = pi_lazy(X, A)
    a = 100000
    b = 100000
    F1 = coarse_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters", 'Coarse feature map (lazy)')
    plt1.show()

    policy = pi_lazy(X, A)
    a = 100000
    b = 100000
    F1 = pl_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters", 'Piecewise feature map (lazy)')
    plt1.show()

    # aggressive policy

    policy = pi_aggressive(X, A)
    a = 100000
    b = 100000
    F1 = fine_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters",
                 'Fine feature map (aggressive)')
    # plt1.title('A Fine feature map')
    plt1.show()

    policy = pi_aggressive(X, A)
    a = 100000
    b = 100000
    F1 = coarse_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters",
                 'Coarse feature map (aggressive)')
    # plt1.title('Coarse feature map')
    plt1.show()

    policy = pi_aggressive(X, A)
    a = 100000
    b = 100000
    F1 = pl_feature(X)
    V1 = TD0(policy, F1, a, b, maxiter=int(1e+4))
    V2 = TD0(policy, F1, a, b, maxiter=int(1e+5))
    V3 = TD0(policy, F1, a, b, maxiter=int(1e+6))
    V4 = TD0(policy, F1, a, b, maxiter=int(1e+7))

    plt1 = line4(V1, V2, V3, V4, "1e+4 iters", "1e+5 iters", "1e+6 iters", "1e+7 iters",
                 'Piecewise feature map (aggressive)')
    plt1.show()

