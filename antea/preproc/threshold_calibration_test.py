import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np


from antea.preproc.threshold_calibration import filter_df_evts

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
