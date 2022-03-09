import pandas as pd
import numpy  as np

from antea.preproc.tdc_corrections import correct_tfine_wrap_around
from antea.preproc.tdc_corrections import compute_tcoarse_wrap_arounds

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
