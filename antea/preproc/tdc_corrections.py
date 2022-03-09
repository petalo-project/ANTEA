import pandas as pd
import numpy  as np

def correct_tfine_wrap_around(df):
    '''
    Corrects the tfine values
    '''
    df['tfine'] = (df['tfine'] + 14) % 1024
