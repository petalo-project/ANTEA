import os
import numpy  as np
import pandas as pd

from pytest import mark

from . import petit_reco_functions as prf

import antea.io.mc_io as mcio


@mark.parametrize("filename data_or_mc".split(),
                  (('petit_mc_test.pet.h5', 'mc'),
                   ('petit_data_test.h5', 'data')))
def test_compute_coincidences(ANTEADATADIR, filename, data_or_mc):
    """
    Checks that both planes of sensors have detected charge.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data_or_mc == 'mc':
        df = mcio.load_mcsns_response(PATH_IN)
    else:
        df = pd.read_hdf(PATH_IN, '/data_0')

    _, evt_groupby = prf.params(df, data_or_mc)
    df_coinc = prf.compute_coincidences(df, data_or_mc)
    sns      = df_coinc.groupby(evt_groupby).sensor_id.unique()
    s_d      = np.array([len(s[s<100]) for s in sns])
    s_c      = np.array([len(s[s>100]) for s in sns])
    assert np.all(s_d) and np.all(s_c)


@mark.parametrize("det_plane central_sns".split(),
                 (( True, prf.central_sns_det),
                  (False, prf.central_sns_coinc)))
def test_select_evts_with_max_charge_at_center(ANTEADATADIR, det_plane, central_sns):
    """
    Checks that the max charge is at center of the chosen plane.
    """
    PATH_IN = os.path.join(ANTEADATADIR, 'petit_mc_test.pet.h5')
    df      = mcio.load_mcsns_response(PATH_IN)

    df_center = prf.select_evts_with_max_charge_at_center(df, det_plane=det_plane)
    if det_plane:
        df_center = df_center[df_center.sensor_id<100]
    else:
        df_center = df_center[df_center.sensor_id>100]
    assert len(df_center) > 0

    for evt in df_center.event_id.unique():
        sns_evt   = df_center[df_center.event_id==evt]
        id_max_ch = sns_evt.loc[sns_evt.charge.idxmax()].sensor_id
        assert id_max_ch in central_sns


@mark.parametrize("filename data_or_mc det_plane variable tot_mode".split(),
                  (('petit_mc_test.pet.h5', 'mc',  True,          'charge', False),
                   ('petit_mc_test.pet.h5', 'mc', False,          'charge', False),
                   ('petit_data_test.h5', 'data',  True, 'efine_corrected', False),
                   ('petit_data_test.h5', 'data', False, 'efine_corrected', False),
                   ('petit_data_test.h5', 'data',  True,          'intg_w', False),
                   ('petit_data_test.h5', 'data',  True,      'intg_w_ToT',  True),
                   ('petit_data_test.h5', 'data', False,      'intg_w_ToT',  True)))
def test_select_evts_with_max_charge_at_center(ANTEADATADIR, filename, data_or_mc,
                                               det_plane, variable, tot_mode):
    """
    Checks that the max charge (in terms of the desired variable) is at center
    of the chosen plane.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data_or_mc == 'mc':
        df = mcio.load_mcsns_response(PATH_IN)
    else:
        df = pd.read_hdf(PATH_IN, '/data_0')
        df['intg_w_ToT'] = df['t2'] - df['t1']

    _, evt_groupby = prf.params(df, data_or_mc)

    if det_plane:
        tofpet_id   = 0
        central_sns = prf.central_sns_det
    else:
        tofpet_id   = 2
        central_sns = prf.central_sns_coinc

    df_center = prf.select_evts_with_max_charge_at_center(df,
                                                          data_or_mc = data_or_mc,
                                                          det_plane  = det_plane,
                                                          variable   = variable,
                                                          tot_mode   = tot_mode)
    df_center = df_center[df_center.tofpet_id==tofpet_id]
    assert len(df_center) > 0

    idx_max = df_center.groupby(evt_groupby)[variable].idxmax()
    for idx in idx_max:
        sns_max = df_center.loc[idx].sensor_id
        assert sns_max in central_sns


@mark.parametrize("filename data_or_mc variable".split(),
                  (('petit_mc_test.pet.h5', 'mc',          'charge'),
                   ('petit_data_test.h5', 'data', 'efine_corrected'),
                   ('petit_data_test.h5', 'data',          'intg_w'),
                   ('petit_data_test.h5', 'data',      'intg_w_ToT')))
def test_contained_evts_in_det_plane_and_compute_ratio_in_corona(ANTEADATADIR, filename, data_or_mc, variable):
    """
    Checks whether the event is fully contained in the detection plane and
    checks that the ratio of charge in the external corona is correct.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data_or_mc == 'mc':
        df = mcio.load_mcsns_response(PATH_IN)
    else:
        df = pd.read_hdf(PATH_IN, '/data_0')

    df_cov  = prf.select_contained_evts_in_det_plane(df, data_or_mc=data_or_mc)
    assert len(np.intersect1d(df_cov.sensor_id.unique(), prf.corona))==0

    ratio_cor = prf.compute_charge_ratio_in_corona(df_cov, data_or_mc=data_or_mc, variable=variable)
    assert np.count_nonzero(ratio_cor.values)==0

    ratios = prf.compute_charge_ratio_in_corona(df, data_or_mc=data_or_mc, variable=variable).values
    assert np.logical_and(ratios >= 0, ratios <= 1).all()
