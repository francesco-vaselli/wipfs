import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def conditioning_plot(reco, samples, gen, target_col, condition_col, *args, **kwargs):

    full = reco[target_col].values
    flash = samples[target_col].values
    conditioning = gen[condition_col].values.astype(bool)

    # Mask (does it change the effective values of dataframes?)

    full = full[conditioning]
    full = full[~np.isnan(full)]
    flash = flash[conditioning]
    flash = flash[~np.isnan(flash)]

    fig = plt.figure()
    plt.hist(full, histtype="step", label="FullSim", ls="--", *args, **kwargs)
    plt.hist(flash, histtype="step", label="FlashSim", *args, **kwargs)
    plt.legend()
    plt.title(f"{target_col}/{condition_col}")

    return fig


if __name__ == "__main__":

    np.random.seed(0)

    N = 10
    x = np.random.rand(N)
    x = pd.DataFrame(data=x, columns=["a"])

    y = np.random.rand(N)
    y = pd.DataFrame(data=y, columns=["a"])

    g = np.random.randint(0, 2, size=N)
    g = pd.DataFrame(data=g, columns=["b"])


    fig = conditioning_plot(x, y, g, "a", "b", bins=10, range=[0, 0.5])
    fig.show()
    