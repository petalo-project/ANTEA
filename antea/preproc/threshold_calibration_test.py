import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np
import os


from antea.preproc.threshold_calibration import filter_df_evts
from antea.preproc.threshold_calibration import get_run_control
from antea.preproc.threshold_calibration import compute_limit_evts_based_on_run_control

def test_filter_df_evts():
    '''
    Check the filtered dataframe by event number
    '''

    df = pd.DataFrame({'evt_number': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                      4, 4, 4, 4, 5, 5, 5, 5],
                       'channel_id': [7, 13, 16, 29, 33, 37, 42, 45, 49, 51, 55,
                                      2, 0, 60, 63, 45, 29, 13, 17, 48, 22, 37]})

    df_filtered = filter_df_evts(df, 2, 3)
    df_expected = df.copy()
    indx        = df_expected[df_expected['evt_number'] != 2].index
    df_expected = df_expected.drop(indx)

    np.testing.assert_array_equal(df_filtered, df_expected)


def test_get_run_control(output_tmpdir):
    '''
    Check the data concatenation of different files and the addition of a new
    column in the last df
    '''
    fname_1 = os.path.join(output_tmpdir, 'test_1_calib.h5')
    fname_2 = os.path.join(output_tmpdir, 'test_2_calib.h5')
    files   = [fname_1, fname_2]

    first_df  = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0, 0, 0],
                              'channel_id' : [4, 17, 29, 33, 57, 16],
                              'tac_id'     : [0, 1, 0, 2, 3, 0],
                              'run_control': [0, 0, 0, 1, 1, 1]})

    second_df = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0, 0, 0],
                              'channel_id' : [5, 16, 24, 44, 58, 13],
                              'tac_id'     : [0, 1, 2, 0, 2, 0],
                              'run_control': [0, 0, 0, 1, 1, 1]})

    first_df['run_control']  = first_df['run_control'].astype(np.uint8)
    second_df['run_control'] = second_df['run_control'].astype(np.uint8)

    first_df.to_hdf (fname_1, key = 'dateEvents', format = 'table')
    second_df.to_hdf(fname_2, key = 'dateEvents', format = 'table')

    df          = get_run_control(files)
    df_expected = pd.concat([first_df, second_df])
    df_expected['fileno'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    df_expected['diff']   = [0, 0, 0, 1, 0, 0, 255, 0, 0, 1, 0, 0]
    np.testing.assert_array_equal(df, df_expected)


def test_compute_limit_evts_based_on_run_control():
    '''
    Check the correct limit obtantion for each group of data based on
    run control number.
    '''
    df = pd.DataFrame({'evt_number' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                       'run_control': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                                       0, 0, 0, 0, 0, 1, 1],
                       'fileno'     : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    df['run_control'] = df['run_control'].astype(np.uint8)
    df['diff'] = df['run_control'].diff().fillna(0)

    df_limits = compute_limit_evts_based_on_run_control(df)
    df_expected = pd.DataFrame({'start': [0, 4, 7, 10, 14, 19],
                                'end'  : [4, 7, 10, 14, 19, 21],
                                'file1': [0, 0, 0, 0, 1, 1],
                                'file2': [0, 0, 0, 1, 1, 1]})

    np.testing.assert_array_equal(df_limits, df_expected)
