import os
import numpy                 as np
import pandas                as pd
import hypothesis.strategies as st

from hypothesis  import given

import tof_functions as tf
from antea.io.mc_io import load_mcTOFsns_response


alfa = st.floats(min_value=1, max_value=100000)
beta = st.floats(min_value=2, max_value=100000)
l    = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)
@given(alfa, beta, l)
def test_spe_dist(alfa, beta, l):
    if alfa == beta:
        return
    exp_dist = tf.spe_dist((alfa, beta), np.unique(l))

    assert len(exp_dist) == len(np.unique(l))
    assert (exp_dist >= 0.).all()



