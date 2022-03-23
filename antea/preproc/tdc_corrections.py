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


def compute_extended_tcoarse(df):
    '''
    Calculates the global tcoarse for each row of data taking into account the
    original tcoarse and the number of tcoarse loops.
    '''
    return df['tcoarse'] + df['nloops'] * 2**16


def add_tcoarse_extended_to_df(df):
    '''
    Adds the calculated tcoarse extended to the original dataframe and deletes
    the tcoarse difference and the number of loops.
    '''
    df['tcoarse']          = df.tcoarse.astype(np.int32)
    df['tcoarse_diff']     = df.tcoarse.diff()
    df['nloops']           = compute_tcoarse_nloops_per_event(df)
    df['tcoarse_extended'] = compute_extended_tcoarse(df)
    df.drop(columns=['tcoarse_diff', 'nloops'], inplace=True)


def compute_integration_window_size(df):
    '''
    It calculates the integration window size taking into account the difference
    number of bits in ecoarse and tcoarse
    '''
    df['intg_w'] = (df.ecoarse - (df.tcoarse % 2**10)).astype('int16')
    df.loc[df['intg_w'] < 0, 'intg_w'] += 2**10
