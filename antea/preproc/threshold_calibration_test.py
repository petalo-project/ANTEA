import pandas           as pd
import numpy            as np
import os

from antea.preproc.threshold_calibration import filter_df_evts
from antea.preproc.threshold_calibration import get_run_control
from antea.preproc.threshold_calibration import compute_limit_evts_based_on_run_control
from antea.preproc.threshold_calibration import process_df
from antea.preproc.threshold_calibration import compute_max_counter_value_for_each_config
from antea.preproc.threshold_calibration import process_run
from antea.preproc.threshold_calibration import find_threshold


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
    Check that each group obtains the correct limit based on the run
    control number
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


def test_process_df():
    '''
    Check that the correct results are obtained using the statistical
    operations for each group of data.
    '''

    df       = pd.DataFrame({'channel_id' : [13, 13, 13, 13, 13, 13, 14, 14,
                                             14, 14, 14],
                             'count'      : [3000, 3800, 4305, 4152, 3200, 3460,
                                             100, 213, 150, 175, 210]})

    field    = 'vth_t1'
    channels = list(range(64))
    value   = 28

    df_tmp      = process_df(df, channels, field, value)
    df_expected = pd.DataFrame({'channel_id': [13, 14],
                                'count'     : [6, 5],
                                'mean'      : [3652.83, 169.6],
                                'std'       : [522.413, 46.832],
                                'min'       : [3000, 100],
                                'max'       : [4305, 213],
                                'sum'       : [21917, 848],
                                'vth_t1'    : [28, 28]})

    assert np.allclose(df_tmp, df_expected, atol = 0.001)


def test_compute_max_counter_value_for_each_config(output_tmpdir):
    '''
    Check that the correct results are obtained using statistical operations and
    the tofpet events array for each group of data.
    '''
    tofpet_id = 0
    field     = 'vth_t1'
    channels  = list(np.arange(64))
    limits    = pd.DataFrame({'start': [0, 4, 7, 10, 14, 19],
                              'end'  : [4, 7, 10, 14, 19, 21],
                              'file1': [0, 0, 0, 0, 1, 1],
                              'file2': [0, 0, 0, 1, 1, 1]})

    fname_1 = os.path.join(output_tmpdir, 'test_3_calib.h5')
    fname_2 = os.path.join(output_tmpdir, 'test_4_calib.h5')
    files   = [fname_1, fname_2]

    df_1    = pd.DataFrame({'evt_number' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'run_control': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                            'tofpet_id'  : [0, 0, 2, 0, 0, 0, 2, 2, 2, 2],
                            'channel_id' : [14, 14, 64, 35, 35, 35, 55, 39, 24, 13],
                            'count'      : [3000, 3200, 2400, 10000, 11000,
                                            10500, 9000, 8000, 9500, 16000]})

    df_2    = pd.DataFrame({'evt_number' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                            'run_control': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                            'tofpet_id'  : [0, 0, 0, 2, 0, 0, 2, 2, 0, 0],
                            'channel_id' : [60, 60, 60, 48, 42, 42, 51, 25, 1, 1],
                            'count'      : [30000, 35000, 33000, 40000, 42000,
                                            43000, 41000, 39000, 60000, 63000]})

    df_1.to_hdf(fname_1, key = 'counter', format = 'table')
    df_2.to_hdf(fname_2, key = 'counter', format = 'table')

    df, tofpet_evts = compute_max_counter_value_for_each_config(tofpet_id, field, channels, files, limits)


    df_expected = pd.DataFrame({'channel_id': [14, 35, 60, 42, 1],
                                'count'     : [2, 3, 3, 2, 2],
                                'mean'      : [3100, 10500, 32666.67, 42500, 61500],
                                'std'       : [141.421, 500, 2516.611, 707.106, 2121.320],
                                'min'       : [3000, 10000, 30000, 42000, 60000],
                                'max'       : [3200, 11000, 35000, 43000, 63000],
                                'sum'       : [6200, 31500, 98000, 85000, 123000],
                                'vth_t1'    : [0, 1, 3, 4, 5]})

    tofpet_evts_expected = np.array([2, 3, 0, 3, 2, 2])

    assert np.allclose(df, df_expected, atol = 0.001)
    np.testing.assert_array_equal(tofpet_evts, tofpet_evts_expected)


def test_process_run(output_tmpdir):
    '''
    Check it returns the correct df with the necessary arithmetic operations
    for each channel and vth value.
    '''
    run_number = 11216
    #Recreate data folder structure
    new_folder_pattern = os.path.join(output_tmpdir, 'analysis/{run}/hdf5/data/')
    new_folder         = new_folder_pattern.format(run = run_number)
    os.makedirs(new_folder)

    #Create HDF5 files with file names unsorted
    sample_dir_5 = os.path.join(new_folder, 'data_0005.h5')
    sample_dir_3 = os.path.join(new_folder, 'data_0003.h5')

    df_1_count  = pd.DataFrame({'evt_number' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'tofpet_id'  : [0, 0, 2, 0, 0, 0, 2, 2, 2, 2],
                          'channel_id' : [14, 14, 64, 35, 35, 35, 55, 39, 24, 13],
                          'run_control': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                          'count'      : [3000, 3200, 2400, 10000, 11000,
                                          10500, 9000, 8000, 9500, 16000]})

    df_1_time = df_1_count.copy()
    df_1_time = df_1_time.drop(['count'], axis = 1)
    df_1_time['timestamp'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*10**6


    df_2_count = pd.DataFrame({'evt_number' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         'tofpet_id'  : [0, 0, 0, 2, 0, 0, 2, 2, 0, 0],
                         'channel_id' : [60, 60, 60, 48, 42, 42, 51, 25, 1, 1],
                         'run_control': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                         'count'      : [30000, 35000, 33000, 40000, 42000,
                                         43000, 41000, 39000, 60000, 63000]})

    df_2_time = df_2_count.copy()
    df_2_time = df_2_time.drop(['count'], axis = 1)
    df_2_time['timestamp'] = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])*10**6


    df_1_count['run_control'] = df_1_count['run_control'].astype(np.uint8)
    df_1_time ['run_control'] = df_1_time ['run_control'].astype(np.uint8)
    df_2_count['run_control'] = df_2_count['run_control'].astype(np.uint8)
    df_2_time ['run_control'] = df_2_time ['run_control'].astype(np.uint8)

    df_1_count.to_hdf(sample_dir_3, key = 'counter',       format = 'table')
    df_1_time. to_hdf(sample_dir_3, key = 'dateEvents', format = 'table')
    df_2_count.to_hdf(sample_dir_5, key = 'counter',       format = 'table')
    df_2_time. to_hdf(sample_dir_5, key = 'dateEvents', format = 'table')

    tofpet_id = 0
    nbits = 22
    field     = 'vth_t1'
    channels  = list(np.arange(64))

    #Check the result
    df_counts   = process_run(run_number, nbits, field, tofpet_id, channels, plot = False, folder=new_folder_pattern)
    df_expected = pd.DataFrame({'channel_id': [14, 35, 60, 42, 1],
                                'count'     : [2, 3, 3, 2, 2],
                                'mean'      : [3100, 10500, 32666.67, 42500, 61500],
                                'std'       : [141.421, 500, 2516.611, 707.106, 2121.320],
                                'min'       : [3000, 10000, 30000, 42000, 60000],
                                'max'       : [3200, 11000, 35000, 43000, 63000],
                                'sum'       : [6200, 31500, 98000, 85000, 123000],
                                'vth_t1'    : [0, 1, 3, 4, 5]})

    assert np.allclose(df_counts, df_expected, atol = 0.001)


def test_find_threshold():
    '''
    Check that the correct results of threshold vth_t1 and vth_t2 are obtained
    for each channel.
    '''
    channels = [0]*64 + [1]*64

    df_1               = pd.DataFrame()
    df_1['channel_id'] = channels

    #Create a list of max values for all channels
    first       = [0]*25
    max_1       = [4, 100, 180, 200, 250, 500, 3000, 10000, 20000, 30000,
                   40000, 50000, 60000, 63000]
    max_2       = max_1.copy()
    max_2[0]    = 12
    max_2[2]    = 182
    last        = [2**16] * 25
    max_counts  = list(np.concatenate((first, max_1, last, first, max_2, last)))
    df_1['max'] = max_counts

    vth = list(range(64))*2
    df_1['vth_t1'] = vth

    df_2 = df_1.copy()
    df_2 = df_2.rename({'vth_t1': 'vth_t2', 'max': 'expected_rate'}, axis=1)

    #Create a list of expected rate values for all channels
    expect_1      = [50, 200, 3000, 10000, 20000, 40000, 60000, 80000, 100000,
                     130000, 170000, 220000, 260000, 290000]
    expect_2      = expect_1.copy()
    expect_2[0]   = 40
    expect_2[1]   = 180
    last_expect   = [300000]*25
    expected_rate = list(np.concatenate((first, expect_1, last_expect, first,
                                         expect_2, last_expect)))

    df_2['expected_rate'] = expected_rate

    threshold_1          = find_threshold(df_1, 22, 'vth_t1')
    threshold_2          = find_threshold(df_2, 22, 'vth_t2', 182)
    expected_threshold_1 = list([25, 24]) + list([0] * 62)
    expected_threshold_2 = list([26, 27]) + list([0] * 62)

    np.testing.assert_array_equal(threshold_1, expected_threshold_1)
    np.testing.assert_array_equal(threshold_2, expected_threshold_2)
