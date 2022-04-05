import pandas as pd
import numpy  as np

from antea.preproc.tdc_corrections import compute_integration_window_size

def correct_efine_wrap_around(df):
    '''
    Corrects the efine value
    '''
    df['efine'] = (df['efine'] + 14) % 1024


def apply_qdc_autocorrection(df, df_qdc):
    '''
    Corrects the efine value subtracting the offset value obtained with
    the calibration in df_qdc
    '''
    df = df.merge(df_qdc, on=['tofpet_id', 'channel_id', 'tac_id', 'intg_w'])
    return (df['efine'] - df.offset)
