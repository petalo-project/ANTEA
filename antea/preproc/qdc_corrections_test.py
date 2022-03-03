import pandas as pd
import numpy  as np

from antea.preproc.qdc_corrections import correct_efine_wrap_around


def test_correct_efine_wrap_around():
    ''' Check the efine correction '''
    df              = pd.DataFrame({'efine':[1,20,300,450,1020]})
    expected_result = np.array([15, 34, 314, 464, 10])

    correct_efine_wrap_around(df)

    np.testing.assert_array_equal(df['efine'].values, expected_result)
