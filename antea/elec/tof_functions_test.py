import os
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis  import given

import tof_functions as tf
from antea.io.mc_io import load_mcTOFsns_response


a = st.floats(min_value=1, max_value=100000)
b = st.floats(min_value=2, max_value=100000)
l = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(a, b, l)
def test_spe_dist(a, b, l):
    """
    This test checks that the function spe_dist returns an array with the distribution value for each time and in case the distribution contains only zeros, it ensures that it returns a fully zeros array, avoiding the normalization.
    """
    l        = np.array(l)
    exp_dist = tf.spe_dist((a, b), np.unique(l))
    count_a  = np.count_nonzero(np.exp(-(1/a)*l))
    count_b  = np.count_nonzero(np.exp(-(1/b)*l))

    if np.isclose(1/a, 1/b, rtol=1e-2) or (not count_a and not count_b):
        assert np.count_nonzero(exp_dist) == 0
        assert np.isclose(np.sum(exp_dist), 0)
    else:
        assert len(exp_dist) == len(np.unique(l))
        assert (exp_dist >= 0.).all()
        assert np.isclose(np.sum(exp_dist), 1)


t = st.tuples(st.floats(min_value=1, max_value=1000), st.floats(min_value=2, max_value=1000))
s = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(t, l, s)
def test_convolve_tof(t, l, s):
    """
    Check that the function convolve_tof returns an array with the adequate length, and, in case it is
    """
    spe_response = tf.spe_dist(t, np.unique(l))
    conv_res     = tf.convolve_tof(spe_response, np.array(s))
    assert len(conv_res) == len(spe_response) + len(s) - 1
    if np.isclose(1/t[0], 1/t[1], rtol=1e-2):
        assert np.sum(conv_res) == 0
    else:
        assert np.isclose(np.sum(s), np.sum(conv_res))
