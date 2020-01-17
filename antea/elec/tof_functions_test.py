import os
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis     import given
from antea.io.mc_io import load_mcTOFsns_response
from antea.elec     import tof_functions   as tf


l = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l)
def test_spe_dist(l):
    """
    This test checks that the function spe_dist returns an array with the distribution value for each time.
    """
    l        = np.array(l)
    exp_dist = tf.spe_dist(np.unique(l))

    assert len(exp_dist) == len(np.unique(l))
    assert (exp_dist >= 0.).all()
    assert np.isclose(np.sum(exp_dist), 1)


s = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l, s)
def test_convolve_tof(l, s):
    """
    Check that the function convolve_tof returns an array with the adequate length, and, in case the array is not empty, checks that the convoluted signal is normalizated to the initial signal.
    """
    spe_response, norm = tf.spe_dist(t, np.unique(np.array(l)))
    conv_res           = tf.convolve_tof(spe_response, np.array(s))
    assert len(conv_res) == len(spe_response) + len(s) - 1
    if np.count_nonzero(spe_response):
        assert np.isclose(np.sum(s), np.sum(conv_res))


def test_tdc_convolution(ANTEADATADIR):
    """
    Check that the function tdc_convolution returns a table with the adequate dimentions and in case the tof dataframe is empty, checks that the table only contains zeros.
    """
    PATH_IN        = os.path.join(ANTEADATADIR, 'ring_test_1000ev.h5')
    tof_response   = load_mcTOFsns_response(PATH_IN)
    SIPM           = {'n_sipms':3500, 'first_sipm':1000, 'tau_sipm':[100,15000]}
    n_sipms        = SIPM['n_sipms']
    first_sipm     = SIPM['first_sipm']
    tau_sipm       = SIPM['tau_sipm']
    TE_range       = [0.25]
    TE_TDC         = TE_range[0]
    time_window    = 10000
    time_bin       = 5
    time           = np.arange(0, 80000, time_bin)
    spe_resp, norm = tf.spe_dist(tau_sipm, time)
    tdc_conv_table = tf.tdc_convolution(tof_response, spe_resp, time_window, n_sipms, first_sipm, TE_TDC)
    assert tdc_conv_table.shape == (time_window + len(spe_resp)-1, n_sipms)

    keys                 = np.array(['event_id', 'sensor_id', 'time_bin', 'charge'])
    wf_df                = pd.DataFrame({}, columns=keys)
    tdc_conv_table_zeros = tf.tdc_convolution(wf_df, spe_resp, time_window, n_sipms, first_sipm, TE_TDC)
    assert np.all(tdc_conv_table_zeros==0)


l1 = st.lists(st.floats(min_value=0, max_value=1000), min_size=2, max_size=100)
l2 = st.lists(st.floats(min_value=0, max_value=1000), min_size=2, max_size=100)
e  = st.floats(min_value=0, max_value=1000)
f  = st.floats(min_value=0, max_value=1000)

@given(l1, l2, e, f)
def test_translate_charge_matrix_to_wf_df(l1, l2, e, f):
    l1    = np.array(l1)
    l2    = np.array(l2)
    col   = np.reshape(l1, (l1.shape[0], 1))
    row   = np.reshape(l2, (1, l2.shape[0]))
    matrx = col*row
    wf_df = tf.translate_charge_matrix_to_wf_df(e, matrx, f)
    assert len(wf_df) == np.count_nonzero(matrx)
    assert len(wf_df.keys()) == 4
    if np.count_nonzero(matrx) == 0:
        assert wf_df.empty
