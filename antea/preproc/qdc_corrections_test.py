import pandas as pd
import numpy  as np

from antea.preproc.qdc_corrections import correct_efine_wrap_around
from antea.preproc.qdc_corrections import apply_qdc_autocorrection


def test_correct_efine_wrap_around():
    ''' Check the efine correction '''
    df              = pd.DataFrame({'efine':[1,20,300,450,1020]})
    expected_result = np.array([15, 34, 314, 464, 10])

    correct_efine_wrap_around(df)

    np.testing.assert_array_equal(df['efine'].values, expected_result)


def test_apply_qdc_autocorrection():
    '''
    Check the result of the efine  correction due to the offset
    '''
    df     = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                           'channel_id': [4, 10, 13, 25, 29, 36, 42, 47, 58, 63, 42, 58],
                           'tac_id'    : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0],
                           'intg_w'    : [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 600, 800],
                           'efine'     : [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]})


    df_qdc = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                           'channel_id': [4,10, 13, 25, 29, 36, 42, 47, 58, 63 ],
                           'tac_id'    : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                           'intg_w'    : [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                           'offset'    : [0, 5, 30, 70, 100, 20, 200, 260, 300, 410]})

    diff            = apply_qdc_autocorrection(df, df_qdc).values
    expected_result = np.array([0, 45, 70, 80, 100, 230, 100, 300, 90, 100, 250, 40])

    np.testing.assert_array_equal(diff, expected_result)
