
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt


def plot_stretched_ring(filename, event=0, sipms_per_row=204, n_rows=16):

    n_sipms = sipms_per_row * n_rows
    data = pd.read_hdf(filename, key='MC').values

    x = np.array(range(n_rows))
    y = np.array(range(sipms_per_row))

    x_hist, y_hist = np.meshgrid(y, x) ## write x and y to be given to hist2d

    x_entries = np.array(x_hist.flatten() + 0.5, dtype='float32')
    y_entries = np.array(y_hist.flatten() + 0.5, dtype='float32')

    h, xedges, yedges, _ = plt.hist2d(x_entries, y_entries, bins=(sipms_per_row, n_rows),
                                      weights = data[event])

    return h, plt.colorbar()
