
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt


def build_xy(sipms_per_row=204, n_rows=16):

    x = np.array(range(n_rows))
    y = np.array(range(sipms_per_row))

    x_hist, y_hist = np.meshgrid(y, x) ## write x and y to be given to hist2d

    ## Use the centers of the bins, instead of the edges
    x_entries = np.array(x_hist.flatten() + 0.5, dtype='float32')
    y_entries = np.array(y_hist.flatten() + 0.5, dtype='float32')

    return x_entries, y_entries


def plot_stretched_ring(filename, event=0, sipms_per_row=204, n_rows=16):

    data = pd.read_hdf(filename, key='MC').values
    x_entries, y_entries = build_xy(sipms_per_row, n_rows)
    h, _, _, _ = plt.hist2d(x_entries, y_entries, bins=(sipms_per_row, n_rows),
                                      range=((0, sipms_per_row), (0, n_rows)),
                                      weights = data[event])

    return h, plt.colorbar()

def plot_stretched_n_rolled_ring(filename, event=0, step_to_roll=0, sipms_per_row=204, n_rows=16):

    data = pd.read_hdf(filename, key='MC').values
    x_entries, y_entries = build_xy(sipms_per_row, n_rows)
    h, _, _ = np.histogram2d(x_entries, y_entries, bins=(sipms_per_row, n_rows),
                             range=((0, sipms_per_row), (0, n_rows)),
                             weights = data[event])

    h_roll = np.roll(h, step_to_roll, axis=0)
    h_T = np.transpose(h_roll)

    h_roll, _, _, _ = plt.hist2d(x_entries, y_entries, bins=(sipms_per_row, n_rows),
                            range=((0, sipms_per_row), (0, n_rows)),
                            weights = h_T.flatten())

    return h_roll, plt.colorbar()
