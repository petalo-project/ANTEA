import os
import pandas as pd
import numpy  as np

from .. database import load_db as db

from .. io.mc_io         import load_mcsns_response
from .. io.mc_io         import load_mcTOFsns_response
from .  sensor_functions import apply_charge_fluctuation
from .  sensor_functions import apply_sipm_pde
from .  sensor_functions import apply_sipm_saturation

import hypothesis.strategies as st
from hypothesis  import given
from typing      import Sequence

def test_number_of_sensors_is_the_same(ANTEADATADIR):
    """
    Checks that the sensors after the fluctuation are the same.
    """

    DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    PATH_IN       = os.path.join(ANTEADATADIR, 'full_body_coinc.h5')
    sns_response  = load_mcsns_response(PATH_IN)
    events        = sns_response.event_id.unique()

    evt = events[0]
    sns_response = sns_response[sns_response.event_id == evt]

    fluct_sns_response = apply_charge_fluctuation(sns_response, DataSiPM_idx)

    original   = sns_response      .groupby(['event_id']).sensor_id.nunique()
    fluctuated = fluct_sns_response.groupby(['event_id']).sensor_id.nunique()

    pd.testing.assert_series_equal(original, fluctuated)


pde = st.floats(min_value=0, max_value=1)
@given(pde)
def test_detected_charge_is_not_greater_than_original(ANTEADATADIR, pde):
    """
    Checks that the charge after applying the pde is never
    higher than the original charge.
    """

    PATH_IN          = os.path.join(ANTEADATADIR, 'ring_test.h5')
    sns_response     = load_mcsns_response(PATH_IN)
    det_sns_response = apply_sipm_pde(sns_response, pde)

    events = sns_response.event_id.unique()
    for evt in events:
        sns_evt     = sns_response[sns_response.event_id == evt]
        det_sns_evt = det_sns_response[det_sns_response.event_id == evt]
        sum_sns     = sns_evt.charge.sum()
        det_sns     = det_sns_evt.charge.sum()

        assert det_sns <= sum_sns
        
        
def test_apply_sipm_saturation(ANTEADATADIR):
    '''
    Checks that the charge after applying the SiPM saturation
    is never higher than the original charge.
    '''
    PATH_IN       = os.path.join(ANTEADATADIR, 'petit_mc_fbk_test.pet.h5')
    sns_response  = load_mcTOFsns_response(PATH_IN)
    events        = sns_response.event_id.unique()

    evt = events[0]
    rec_time = 80
    sns_response = sns_response[sns_response.event_id == evt]

    sat_sns_response = sns_response.groupby(['event_id','sensor_id'], as_index=False).apply(apply_sipm_saturation, rec_time)

    assert  (sat_sns_response.charge.values).tolist() <= (np.ones(len(sat_sns_response.charge))).tolist()

