import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np


def filter_df_evts(df, evt_start, evt_end):
    '''
    Returns a dataframe filtered by a given range of the event numbers
    '''
    evt_filter = (df.evt_number >= evt_start) & (df.evt_number < evt_end)
    df_filtered = df[evt_filter]
    return df_filtered
