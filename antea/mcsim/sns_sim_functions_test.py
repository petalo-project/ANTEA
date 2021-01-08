import os

from .. io.mc_io      import load_mcsns_response
from . import sns_sim_functions as simf

import hypothesis.strategies as st
from hypothesis  import given

pde = st.floats(min_value=0, max_value=1)

@given(pde)
def test_detected_charge_is_not_greater_than_original(ANTEADATADIR, pde):
    """
    Checks that the charge after applying the pde is never
    higher than the original charge.
    """

    PATH_IN          = os.path.join(ANTEADATADIR, 'ring_test.h5')
    sns_response     = load_mcsns_response(PATH_IN)
    det_sns_response = simf.apply_sipm_pde(sns_response, pde)

    events = sns_response.event_id.unique()
    for evt in events[:]:
        sns_evt     = sns_response[sns_response.event_id == evt]
        det_sns_evt = det_sns_response[det_sns_response.event_id == evt]

        sum_sns = sns_evt.charge.sum()
        det_sns = det_sns_evt.charge.sum()

        assert det_sns <= sum_sns
