import pandas as pd
import numpy  as np

def correct_tfine_wrap_around(df):
    '''
    Corrects the tfine values
    '''
    df['tfine'] = (df['tfine'] + 14) % 1024


def compute_tcoarse_wrap_arounds(df):
    '''
    It returns the data position where tcoarse finishes a cycle, and the first
    and last data position
    '''
    limits = df[df.tcoarse_diff < -30000].index #tcoarse cycle change
    first  = df.index[0]
    last   = df.index[-1]
    limits = np.concatenate([np.array([first]), limits.values, np.array([last])])
    return limits


def compute_tcoarse_nloops_per_event(df):
    '''
    It returns an array with the number of loops for each row of data. It is
    calculated per event number
    '''
    limits = df.groupby('evt_number').apply(compute_tcoarse_wrap_arounds)

    nloops = np.zeros(df.shape[0], dtype='int32')

    for evt_limits in limits.values:
        for i in range(evt_limits.shape[0]-1):
            start = evt_limits[i]
            end   = evt_limits[i+1]

            nloops[start:end+1] = i

    return nloops
