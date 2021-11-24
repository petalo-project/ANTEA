import os
import numpy  as np
import pandas as pd

from pytest import mark

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


@mark.parametrize("det_plane variable tot_mode".split(),
                  (( True, 'efine_corrected', False),
                   (False, 'efine_corrected', False),
                   ( True,          'intg_w', False),
                   ( True,      'intg_w_ToT',  True),
                   (False,     'intg_w_ToT',  True)))
def test_select_evts_with_max_charge_at_center(ANTEADATADIR, det_plane, variable, tot_mode):
    """
    Checks that the max charge (in terms of the desired variable) is at center
    of the chosen plane.
    """
    PATH_IN     = os.path.join(ANTEADATADIR, 'data_petbox_test.h5')
    df          = pd.read_hdf(PATH_IN, '/data_0')
    df['intg_w_ToT'] = df['t2'] - df['t1']

    if det_plane:
        tofpet_id   = 0
        central_sns = [44, 45, 54, 55]
    else:
        tofpet_id   = 2
        central_sns = [122, 123, 132, 133]

    df_center = drf.select_evts_with_max_charge_at_center(df,
                                                          det_plane = det_plane,
                                                          variable  = variable,
                                                          tot_mode  = tot_mode)
    df_center = df_center[df_center.tofpet_id==tofpet_id]
    all_max   = df_center.groupby(['evt_number', 'cluster'])[variable].max()
    if len(all_max):
        assert np.all(np.array([df_center[df_center[variable]==m].sensor_id.values
                                in central_sns for m in all_max]))
