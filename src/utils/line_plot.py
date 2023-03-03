import matplotlib.pyplot as plt
import numpy as np


def line4(a, b, c, d, A, B, C, D, E):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 10))
    x = np.arange(100)
    fig.suptitle(E)
    ax1.plot(x, a)
    ax2.plot(x, b)
    ax3.plot(x, c)
    ax4.plot(x, d)
    ax1.title.set_text(A)
    ax2.title.set_text(B)
    ax3.title.set_text(C)
    ax4.title.set_text(D)
    plt.show()

    return plt
