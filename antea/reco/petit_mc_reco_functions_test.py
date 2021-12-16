import os
import numpy  as np
import pandas as pd

from pytest import mark

from . import petit_mc_reco_functions   as pmrf
from . import petit_data_reco_functions as drf

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


@mark.parametrize("det_plane central_sns".split(),
                 (( True, drf.central_sns_det),
                  (False, drf.central_sns_coinc)))
def test_select_evts_with_max_charge_at_center_mc(ANTEADATADIR, det_plane, central_sns):
    """
    Checks that the max charge is at center of the chosen plane.
    """
    PATH_IN = os.path.join(ANTEADATADIR, 'petit_mc_test.pet.h5')
    df      = mcio.load_mcsns_response(PATH_IN)

    df_center = pmrf.select_evts_with_max_charge_at_center_mc(df, det_plane=det_plane)
    if det_plane:
        df_center = df_center[df_center.sensor_id<100]
    else:
        df_center = df_center[df_center.sensor_id>100]
    assert len(df_center) > 0

    for evt in df_center.event_id.unique():
        sns_evt   = df_center[df_center.event_id==evt]
        id_max_ch = sns_evt.loc[sns_evt.charge.idxmax()].sensor_id
        assert id_max_ch in central_sns


def test_contained_evts_in_det_plane_and_compute_percentage_in_corona(ANTEADATADIR):
    """
    Checks whether the event is fully contained in the detection plane and
    checks that the percentage of charge in the external corona is correct.
    """
    PATH_IN = os.path.join(ANTEADATADIR, 'petit_mc_test.pet.h5')
    df      = mcio.load_mcsns_response(PATH_IN)

    df_cov  = pmrf.select_contained_evts_in_det_plane_mc(df)
    assert len(np.intersect1d(df_cov.sensor_id.unique(), drf.corona))==0

    perc_cor = pmrf.compute_charge_percentage_in_corona_mc(df_cov)
    assert np.count_nonzero(perc_cor.values)==0

    percs = pmrf.compute_charge_percentage_in_corona_mc(df).values
    assert np.logical_and(percs >= 0, percs <= 100).all()
