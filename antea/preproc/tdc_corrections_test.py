import pandas as pd
import numpy  as np

from antea.preproc.tdc_corrections import correct_tfine_wrap_around

def test_correct_tfine_wrap_around():
    ''' Check the tfine correction '''
    df              = pd.DataFrame({'tfine':[1, 20, 300, 450, 1020]})
    expected_result = np.array([15, 34, 314, 464, 10])

    correct_tfine_wrap_around(df)

    np.testing.assert_array_equal(df['tfine'].values, expected_result)
