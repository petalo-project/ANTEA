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
    Check that the function convolve_tof returns an array with the adequate length, and, in case it is
    """
    spe_response = tf.spe_dist(np.unique(np.array(l)))
    conv_res     = tf.convolve_tof(spe_response, np.array(s))
    assert len(conv_res) == len(spe_response) + len(s) - 1
    if np.count_nonzero(spe_response):
        assert np.isclose(np.sum(s), np.sum(conv_res))


