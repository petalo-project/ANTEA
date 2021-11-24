import os
import numpy  as np
import pandas as pd

from . import data_reco_functions as drf


def test_compute_coincidences(ANTEADATADIR):
    """
    Checks that both planes of sensors have detected charge.
    """
    PATH_IN  = os.path.join(ANTEADATADIR, 'data_petbox_test.h5')
    df       = pd.read_hdf(PATH_IN, '/data_0')
    df_coinc = drf.compute_coincidences(df)
    sns      = df_coinc.groupby(['evt_number', 'cluster']).sensor_id.unique()
    s_d      = np.array([len(s[s<100]) for s in sns])
    s_c      = np.array([len(s[s>100]) for s in sns])
    assert np.all(s_d) and np.all(s_c)
