import pandas as pd
import numpy  as np

def correct_tfine_wrap_around(df):
    '''
    Corrects the tfine values according to equation 5 in the PETsys datasheet
    (rev 13).
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


def apply_tdc_correction(df, df_tdc, field='tfine'):
    '''
    This function is used to calculate t1 with field = tfine and to calculate
    t2 with field = efine
    '''

    df = df.reset_index().merge(df_tdc[['channel_id', 'tac_id', 'amplitude', 'offset']], on=['channel_id', 'tac_id'])
    df = df.sort_values('index').set_index('index')
    df.index.name = None

    period = 360
    correctd_field = f'{field}_corrected'
    df[correctd_field] = (period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*df.amplitude))*(df[field]-df.offset)))
    df.loc[df[correctd_field] < 0, correctd_field] += period
    df = df.drop(columns=['amplitude', 'offset'])

    if field == 'tfine':
        df['t1'] = df.tcoarse_extended - (360 - df[correctd_field]) / 360
    else: # for efine
        df['t2'] = df.tcoarse_extended + df.intg_w - (360 - df[correctd_field]) / 360
    return df


def apply_tdc_linear_correction(df, df_tdc, field='tfine'):
    '''
    This function is used to calculate t1 with field = tfine and to calculate
    t2 with field = efine using the linear TDC fit.
    '''

    df = df.reset_index().merge(df_tdc[['channel_id', 'tac_id', 'slope_low',
                                        'origin_low', 'slope_high', 'origin_high',
                                        'shift_phase', 'shift_mode']],
                                        on=['channel_id', 'tac_id'])
    df = df.sort_values('index').set_index('index')
    df.index.name = None

    period         = 360
    corr_field     = f'{field}_corrected'
    df[corr_field] = np.array(0.0)

    df_l = df[df[field] <  df['shift_mode']]
    df_h = df[df[field] >= df['shift_mode']]

    df.loc[df_l.index, corr_field] = (df_l[field] - df_l.origin_low) / df_l.slope_low - df_l.shift_phase
    df.loc[df_h.index, corr_field] = (df_h[field] - df_h.origin_high) / df_h.slope_high - df_h.shift_phase
    df.loc[df[corr_field] < 0, corr_field] += 360


    df = df.drop(columns=['slope_low', 'origin_low', 'slope_high', 'origin_high',
                          'shift_phase', 'shift_mode'])

    if field == 'tfine':
        df['t1'] = df.tcoarse_extended - (360 - df[corr_field]) / 360
    else: # for efine
        df['t2'] = df.tcoarse_extended + df.intg_w - (360 - df[corr_field]) / 360
    return df
