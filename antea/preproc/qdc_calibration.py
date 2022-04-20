import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pylab as plt


def plot_all_channels(df, xlim=None):
    '''
    Plots the efine vs the integration window size for each channel and tac id 0
    '''
    for ch in df.channel_id.unique():
        df_filtered = df[(df.channel_id == ch) & (df.tac_id == 0)]
        plt.plot(df_filtered.intg_w, df_filtered.efine)

    if xlim:
        plt.xlim(*xlim)
