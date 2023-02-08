import matplotlib.pyplot as plt
import numpy as np


def plot_3(A, B, C, a, b):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    a = a.reshape((10, 10))
    b = b.reshape((10,10))
    z1 = ax1.imshow(a)
    z2 = ax2.imshow(b)
    #z3 = ax3.imshow(np.abs(a - b))
    z3 = ax3.imshow((a - b))
    ax1.title.set_text(A)
    ax2.title.set_text(B)
    ax3.title.set_text(C)
    plt.colorbar(z1, ax=ax1, fraction=0.046, pad=0.2)
    plt.colorbar(z2, ax=ax2, fraction=0.046, pad=0.2)
    plt.colorbar(z3, ax=ax3, fraction=0.046, pad=0.2)
    plt.show()
    return plt
