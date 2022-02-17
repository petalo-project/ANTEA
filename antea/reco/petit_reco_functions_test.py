import os
import numpy  as np
import pandas as pd

from pytest import mark

from . import petit_reco_functions as prf

import antea.io.mc_io as mcio


@mark.parametrize("filename data".split(),
                  (('petit_mc_test.pet.h5', False),
                   ('petit_data_test.h5',    True)))
def test_compute_coincidences(ANTEADATADIR, filename, data):
    """
    Checks that both planes of sensors have detected charge.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data:
        df          = pd.read_hdf(PATH_IN, '/data_0')
        evt_groupby = ['evt_number', 'cluster']
    else:
        df              = mcio.load_mcsns_response(PATH_IN)
        df['tofpet_id'] = df['sensor_id'].apply(prf.tofpetid)
        evt_groupby     = ['event_id']

    df_coinc = prf.compute_coincidences(df, evt_groupby)
    sns      = df_coinc.groupby(evt_groupby).sensor_id.unique()
    s_d      = np.array([len(s[s<100]) for s in sns])
    s_c      = np.array([len(s[s>100]) for s in sns])
    assert np.all(s_d) and np.all(s_c)


@mark.parametrize("filename data det_plane coinc_plane_1_tile variable tot_mode".split(),
                  (('petit_mc_test.pet.h5', False, True,  True,          'charge', False),
                   ('petit_mc_test.pet.h5', False, True, False,          'charge', False),
                   ('petit_data_test.h5',    True, True,  True, 'efine_corrected', False),
                   ('petit_data_test.h5',    True, True, False, 'efine_corrected', False),
                   ('petit_data_test.h5',    True, True,  True,          'intg_w', False),
                   ('petit_data_test.h5',    True, True,  True,      'intg_w_ToT',  True),
                   ('petit_data_test.h5',    True, True, False,      'intg_w_ToT',  True)))
def test_select_evts_with_max_charge_at_center(ANTEADATADIR, filename, data, det_plane,
                                               coinc_plane_1_tile, variable, tot_mode):
    """
    Checks that the max charge (in terms of the desired variable) is at center
    of the chosen plane.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data:
        df = pd.read_hdf(PATH_IN, '/data_0')
        df['intg_w_ToT'] = df['t2'] - df['t1']
        evt_groupby      = ['evt_number', 'cluster']
    else:
        df = mcio.load_mcsns_response(PATH_IN)
        df['tofpet_id'] = df['sensor_id'].apply(prf.tofpetid)
        evt_groupby     = ['event_id']

    if det_plane:
        tofpet_id   = 0
        central_sns = prf.central_sns_det
    else:
        tofpet_id   = 2
        if coinc_plane_1_tile:
            central_sns = prf.central_sns_coinc_1tile
        else:
            central_sns = prf.central_sns_coinc_4tiles

    df_center = prf.select_evts_with_max_charge_at_center(df,
                                                          evt_groupby        = evt_groupby,
                                                          det_plane          = det_plane,
                                                          coinc_plane_1_tile = coinc_plane_1_tile,
                                                          variable           = variable,
                                                          tot_mode           = tot_mode)
    df_center = df_center[df_center.tofpet_id==tofpet_id]
    assert len(df_center) > 0

    idx_max = df_center.groupby(evt_groupby)[variable].idxmax()
    for idx in idx_max:
        sns_max = df_center.loc[idx].sensor_id
        assert sns_max in central_sns


@mark.parametrize("filename data variable".split(),
                  (('petit_mc_test.pet.h5', False,          'charge'),
                   ('petit_data_test.h5',    True, 'efine_corrected'),
                   ('petit_data_test.h5',    True,          'intg_w'),
                   ('petit_data_test.h5',    True,      'intg_w_ToT')))
def test_contained_evts_in_det_plane_and_compute_ratio_in_corona(ANTEADATADIR, filename, data, variable):
    """
    Checks whether the event is fully contained in the detection plane and
    checks that the ratio of charge in the external corona is correct.
    """
    PATH_IN = os.path.join(ANTEADATADIR, filename)
    if data:
        df               = pd.read_hdf(PATH_IN, '/data_0')
        df['intg_w_ToT'] = df['t2'] - df['t1']
        evt_groupby      = ['evt_number', 'cluster']
    else:
        df = mcio.load_mcsns_response(PATH_IN)
        df['tofpet_id'] = df['sensor_id'].apply(prf.tofpetid)
        evt_groupby     = ['event_id']

    df_cov  = prf.select_contained_evts_in_det_plane(df, evt_groupby=evt_groupby)
    assert len(np.intersect1d(df_cov.sensor_id.unique(), prf.corona))==0

    ratio_cor = prf.compute_charge_ratio_in_corona(df_cov, evt_groupby=evt_groupby, variable=variable)
    assert np.count_nonzero(ratio_cor.values)==0

    ratios = prf.compute_charge_ratio_in_corona(df, evt_groupby=evt_groupby, variable=variable).values
    assert np.logical_and(ratios >= 0, ratios <= 1).all()
