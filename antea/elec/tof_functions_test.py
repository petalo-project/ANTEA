import os
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis     import given
from pytest         import mark
from antea.io.mc_io import load_mcTOFsns_response
from antea.elec     import tof_functions   as tf


l = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l)
def test_apply_spe_dist(l):
    """
    This test checks that the function apply_spe_dist returns an array with the distribution value for each time.
    """
    l = np.array(l)
    exp_dist, norm_dist = tf.apply_spe_dist(np.unique(l))

    assert len(exp_dist) == len(np.unique(l))
    assert (exp_dist >= 0.).all()
    assert np.isclose(np.sum(exp_dist), 1)


@mark.parametrize('time, time_dist',
                  ((  0, 0),
                   (100, 0.65120889),
                   (np.array([1,2,3]), np.array([0.01029012, 0.02047717, 0.03056217]))))
def test_spe_dist(time, time_dist):
    """
    Spe_dist is an analitic function, so this test takes some values and checks that the function returns the correct value for each one.
    """
    result = tf.spe_dist(time)
    assert np.all(result) == np.all(time_dist)


s = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l, s)
def test_convolve_tof(l, s):
    """
    Check that the function convolve_tof returns an array with the adequate length, and, in case the array is not empty, checks that the convoluted signal is normalizated to the initial signal.
    """
    spe_response, norm = tf.apply_spe_dist(np.unique(np.array(l)))
    conv_res           = tf.convolve_tof(spe_response, np.array(s))
    assert len(conv_res) == len(spe_response) + len(s) - 1
    if np.count_nonzero(spe_response):
        assert np.isclose(np.sum(s), np.sum(conv_res))


@mark.parametrize('filename',
                  (('ring_test.h5'),
                   ('full_body_1ev.h5')))
def test_tdc_convolution(ANTEADATADIR, filename):
    """
    Check that the function tdc_convolution returns a table with the adequate dimensions and in case the tof dataframe is empty, checks that the table only contains zeros.
    """
    PATH_IN        = os.path.join(ANTEADATADIR, filename)
    tof_response   = load_mcTOFsns_response(PATH_IN)
    events         = tof_response.event_id.unique()
    te_tdc         = 0.25
    time_window    = 10000
    time_bin       = 5
    time           = np.arange(0, 80000, time_bin)
    spe_resp, norm = tf.apply_spe_dist(time)
    for evt in events:
        evt_tof = tof_response[tof_response.event_id == evt]
        tof_sns = evt_tof.sensor_id.unique()
        for s_id in tof_sns:
            tdc_conv = tf.tdc_convolution(tof_response, spe_resp, s_id, time_window, te_tdc)
            assert len(tdc_conv) == time_window + len(spe_resp) - 1
            if len(tof_response[(tof_response.sensor_id == s_id) &
                            (tof_response.time_bin > time_window)]) == 0:
                assert np.count_nonzero(tdc_conv) > 0


e     = st.integers(min_value=0,     max_value= 1000)
s_id  = st.integers(min_value=-3500, max_value=-1000)
l2    = st.lists(st.floats(min_value=0, max_value=1000), min_size=2, max_size=100)

@given(e, s_id, l2)
def test_translate_charge_conv_to_wf_df(e, s_id, l2):
    """
    Look whether the translate_charge_conv_to_wf_df function returns a dataframe with the same number of rows as the input numpy array and four columns. Three of this columns must contain integers.
    """
    l2    = np.array(l2)
    wf_df = tf.translate_charge_conv_to_wf_df(e, s_id, l2)
    assert len(wf_df) == np.count_nonzero(l2)
    assert len(wf_df.keys()) == 4
    if np.count_nonzero(l2) == 0:
        assert wf_df.empty
    else:
        assert wf_df.event_id .dtype == 'int32'
        assert wf_df.sensor_id.dtype == 'int32'
        assert wf_df.time_bin .dtype == 'int32'
