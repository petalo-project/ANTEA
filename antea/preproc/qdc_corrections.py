import pandas as pd
import numpy  as np

def correct_efine_wrap_around(df):
    '''
    Corrects the efine value
    '''
    df['efine'] = (df['efine'] + 14) % 1024
