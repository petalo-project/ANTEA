import pandas           as pd
import numpy            as np
import os

from antea.preproc.tdc_tfine_calibration import process_df_to_assign_tpulse_delays
from antea.preproc.tdc_tfine_calibration import compute_normalized_histogram



def test_process_df_to_assign_tpulse_delays(output_tmpdir):
    '''
    Check the correct dataframe result, the tofpet events array
    for each group of data and the number of wrong rows values and
    the channel_id they are.
    '''
    limits = pd.DataFrame({'start' : [0, 4, 7, 10, 14, 19],
                           'end'   : [4, 7, 10, 14, 19, 21],
                           'file1' : [0, 0, 0, 0, 1, 1],
                           'file2' : [0, 0, 0, 1, 1, 1]})

    configs = np.tile(np.arange(0, 30000, 10000), 2)

    fname_1 = os.path.join(output_tmpdir, 'test_tdc_calib_1.h5')
    fname_2 = os.path.join(output_tmpdir, 'test_tdc_calib_2.h5')
    files   = [fname_1, fname_2]

    df_1    = pd.DataFrame({'evt_number' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'run_control': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                            'tofpet_id'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'channel_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            'tfine'      : [317, 319, 315, 300, 305, 302,
                                            290, 293, 291, 315]})

    df_2    = pd.DataFrame({'evt_number' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                            'run_control': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                            'tofpet_id'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'channel_id' : [1, 56, 1, 1, 1, 1, 1, 34, 1, 1],
                            'tfine'      : [317, 320, 319, 300, 303, 301, 299, 303,
                                            290, 291]})

    df_1.to_hdf(fname_1, key = 'data', format = 'table')
    df_2.to_hdf(fname_2, key = 'data', format = 'table')

    df, tofpet_evts, wrong_rows, wrong_ch_t = process_df_to_assign_tpulse_delays(files, limits, configs)

    df_2                 = df_2[df_2['channel_id'] == 1]
    df_expected          = pd.concat([df_1, df_2])
    df_expected['delay'] = [0, 0, 0, 10, 10, 10, 20, 20, 20, 0,
                            0, 0, 10, 10, 10, 10, 20, 20]

    tofpet_evts_expected = np.array([3, 3, 3, 3, 4, 2])
    wrong_rows_expected  =  2
    wrong_ch_t_expected  = [56, 34]


    np.testing.assert_array_equal(df, df_expected)
    np.testing.assert_array_equal(tofpet_evts, tofpet_evts_expected)
    np.testing.assert_array_equal(wrong_rows, wrong_rows_expected)
    np.testing.assert_array_equal(wrong_ch_t, wrong_ch_t_expected)


def test_compute_normalized_histogram():
    '''
    Check that the counts and edges obtained from a values
    array given are correct.
    '''
    bins       = 360
    hist_range = np.array([0, 360])
    values     = np.array([200, 201, 201, 202, 202, 202, 203, 203, 203, 203,
                           203, 204, 204, 204, 205, 205, 206])

    counts, xs      = compute_normalized_histogram(values, hist_range, bins)

    counts_expected = np.concatenate((np.tile(np.arange(1), 200), np.array([1, 2,
                      3, 5, 3, 2, 1])/17 , np.tile(np.arange(1), 153)), axis = 0)

    xs_expected     = np.arange(0.5, 360.5, 1)

    assert np.allclose(counts, counts_expected, atol = 0.001)
    assert np.allclose(xs, xs_expected, atol = 0.001)
