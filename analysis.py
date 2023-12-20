import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl


mpl.use('TkAgg')
x, energy = pkl.load(open('ising.pkl', 'rb'))
runs = len(energy)

autocorr = np.empty(runs-1)
for i in range(runs-1):
    autocorr[i] = pd.Series(energy[i:]).autocorr()

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.arange(runs), energy)
axs[1].plot(np.arange(runs-1), autocorr)
plt.show()
