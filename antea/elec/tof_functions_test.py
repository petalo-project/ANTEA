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
    exp_dist = tf.spe_dist((a, b), np.unique(l))
    if np.isclose(1/a, 1/b, atol=1e-2):
        assert np.count_nonzero(exp_dist) == 0
        assert np.isclose(np.sum(exp_dist), 0)
    else:
        assert len(exp_dist) == len(np.unique(l))
        assert (exp_dist >= 0.).all()
        assert np.isclose(np.sum(exp_dist), 1)
