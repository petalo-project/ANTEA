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
