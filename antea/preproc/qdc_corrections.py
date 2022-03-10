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


def compute_qdc_calibration_using_mode(df_tpulse, max_intg_w = 291):
    '''
    It sorts data taking into account tofpet id, channel id and tac id and it
    calculates the integration window size and the corrected efine
    '''
    compute_integration_window_size(df_tpulse)
    correct_efine_wrap_around(df_tpulse)

    df_calib = df_tpulse.groupby(['tofpet_id', 'channel_id', 'tac_id',
                                  'intg_w'])['efine'].agg(lambda x: x.value_counts().index[0])
    df_calib = df_calib.reset_index()

    # Replace first entries (< 5) with 0 to avoid out of range errors
    # when interpolating (really) small windows.
    df_calib.loc[df_calib.intg_w < 5, 'intg_w'] = 0
    df_calib.loc[df_calib.intg_w == 0, 'efine'] = 0
    # Replace last entry (291) with 2000 to avoid out of range errors
    # when interpolating (really) large windows.
    df_calib.loc[df_calib.intg_w == max_intg_w, 'intg_w'] = 2000

    return df_calib
