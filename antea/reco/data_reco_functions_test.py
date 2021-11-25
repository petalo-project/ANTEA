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
                   (False,      'intg_w_ToT',  True)))
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
        central_sns = drf.central_sns_h
    else:
        tofpet_id   = 2
        central_sns = drf.central_sns_f

    df_center = drf.select_evts_with_max_charge_at_center(df,
                                                          det_plane = det_plane,
                                                          variable  = variable,
                                                          tot_mode  = tot_mode)
    df_center = df_center[df_center.tofpet_id==tofpet_id]
    all_max   = df_center.groupby(['evt_number', 'cluster'])[variable].max()
    if len(all_max):
        assert np.all(np.array([df_center[df_center[variable]==m].sensor_id.values
                                in central_sns for m in all_max]))


corona = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
          51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]

def test_contained_evts_in_det_plane_and_compute_percentage_in_corona(ANTEADATADIR):
    """
    Checks whether the event is fully contained in the detection plane and
    checks that the percentage of charge in the external corona is correct.
    """
    PATH_IN  = os.path.join(ANTEADATADIR, 'data_petbox_test.h5')
    df       = pd.read_hdf(PATH_IN, '/data_0')
    df_cov   = drf.select_contained_evts_in_det_plane(df)
    assert len(np.intersect1d(df_cov.sensor_id.unique(), corona))==0

    perc_cor = drf.compute_charge_percentage_in_corona(df_cov)
    assert np.count_nonzero(perc_cor.values)==0

    percs = drf.compute_charge_percentage_in_corona(df).values
    assert np.logical_and(percs >= 0, percs <= 100).all()
