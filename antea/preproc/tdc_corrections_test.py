import pandas as pd
import numpy  as np

from antea.preproc.tdc_corrections import correct_tfine_wrap_around
from antea.preproc.tdc_corrections import compute_tcoarse_wrap_arounds
from antea.preproc.tdc_corrections import compute_tcoarse_nloops_per_event
from antea.preproc.tdc_corrections import compute_extended_tcoarse
from antea.preproc.tdc_corrections import add_tcoarse_extended_to_df
from antea.preproc.tdc_corrections import compute_integration_window_size

def test_correct_tfine_wrap_around():
    ''' Check the tfine correction '''
    df              = pd.DataFrame({'tfine':[1, 20, 300, 450, 1020]})
    expected_result = np.array([15, 34, 314, 464, 10])

    correct_tfine_wrap_around(df)

    np.testing.assert_array_equal(df['tfine'].values, expected_result)


def test_compute_tcoarse_wrap_arounds():
    ''' Check the limits of a new tcoarse cycle '''
    df = pd.DataFrame({'tcoarse': [1500, 15000, 20000, 30500, 32000, 40000, 6000,
                                   6050, 10000, 50000, 1000, 1100, 3000, 10000]})
    df['tcoarse_diff'] = df.tcoarse.diff()
    df = df.fillna(0)

    limits          = compute_tcoarse_wrap_arounds(df)
    expected_result = np.array([0, 6, 10, 13])

    np.testing.assert_array_equal(limits, expected_result)


def test_compute_tcoarse_nloops_per_event():
    '''
    Check the number of tcoarse cycle we have to add to tcoarse taking into account
    the event number and the limits of cycles obtained with tcoarse_diff
    '''
    df = pd.DataFrame({'evt_number'  : [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                       'tcoarse'     : [200, 1000, 15000, 40000, 2000, 30000,
                                        60000, 400, 2000, 35000, 50000, 65000]})

    df['tcoarse_diff'] = df.tcoarse.diff()
    df = df.fillna(0)

    nloops          = compute_tcoarse_nloops_per_event(df)
    expected_result = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1])

    np.testing.assert_array_equal(nloops, expected_result)


def test_compute_extended_tcoarse():
    '''
    Check the addition of tcoarse cycles to original tcoarse
    '''
    df = pd.DataFrame({'tcoarse': [1000, 10000, 60000, 30000, 40000, 65000, 40000, 65000],
                       'nloops': [ 0, 0, 0, 1, 1, 1, 2, 2]})

    extended_tcoarse = compute_extended_tcoarse(df).values
    expected_result  = np.array([1000, 10000, 60000, 95536, 105536, 130536, 171072, 196072])

    np.testing.assert_array_equal(extended_tcoarse, expected_result)


def test_add_tcoarse_extended_to_df():
    '''
    Check the addition of a new column (tcoarse extended) to the original dataframe
    '''
    df = pd.DataFrame({'evt_number'  : [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                       'tcoarse'     : [200, 1000, 15000, 40000, 2000, 30000,
                                        60000, 400, 2000, 35000, 50000, 65000]})

    df_expected = df.copy()
    df_expected['tcoarse_extended'] = [200, 1000, 15000, 40000, 67536, 95536,
                                       60000, 65936, 67536, 100536, 115536, 130536]
    add_tcoarse_extended_to_df(df)

    np.testing.assert_array_equal(df, df_expected)


def test_compute_integration_window_size():
    '''
    Check the integration window calculation
    '''
    df = pd.DataFrame({'ecoarse' : [10, 200, 400, 700, 1000, 1024, 100],
                       'tcoarse' : [1000, 10000, 20000, 35000, 40000, 63000, 200]})

    compute_integration_window_size(df)
    df_expected           = df.copy()
    df_expected['intg_w'] = [34, 440, 880, 516, 936, 488, 924]

    np.testing.assert_array_equal(df, df_expected)
