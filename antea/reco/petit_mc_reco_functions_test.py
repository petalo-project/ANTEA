import os
import numpy  as np
import pandas as pd

from pytest import mark

from . import petit_mc_reco_functions as pmrf

import antea.io.mc_io as mcio


def test_compute_coincidences_mc(ANTEADATADIR):
    """
    Checks that both planes of sensors have detected charge.
    """
    PATH_IN  = os.path.join(ANTEADATADIR, 'petit_mc_test.pet.h5')
    df       = mcio.load_mcsns_response(PATH_IN)
    df_coinc = pmrf.compute_coincidences_mc(df)
    sns      = df_coinc.groupby('event_id').sensor_id.unique()
    s_d      = np.array([len(s[s<100]) for s in sns])
    s_c      = np.array([len(s[s>100]) for s in sns])
    assert np.all(s_d) and np.all(s_c)
