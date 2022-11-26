import matplotlib.pyplot as plt
import numpy as np

from proclc.smoothing_spline import SplineSmoothing


def load_curve():
    x = np.linspace(-10, 10, num=100)
    y = x * x + np.random.randn(x.size) * 5

    smoother = SplineSmoothing(criteria="gcv")
    yf = smoother.fit(x, y)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, ".")
    ax.plot(x, yf, "r-")
    plt.show()


if __name__ == "__main__":
    load_curve()
