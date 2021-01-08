import os
import pandas as pd

from .. database import load_db as db

from .. io.mc_io         import load_mcsns_response
from .  sensor_functions import apply_charge_fluctuation

def test_number_of_sensors_is_the_same(ANTEADATADIR):
    """
    Checks that the sensors after the fluctuation are the same.
    """

    DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    PATH_IN       = os.path.join(ANTEADATADIR, 'full_body_1ev.h5')
    sns_response  = load_mcsns_response(PATH_IN)
    events        = sns_response.event_id.unique()

    fluct_sns_response = apply_charge_fluctuation(sns_response, DataSiPM_idx)

    original   = sns_response.groupby(['event_id']).sensor_id.nunique()
    fluctuated = fluct_sns_response.groupby(['event_id']).sensor_id.nunique()

    pd.testing.assert_series_equal(original, fluctuated)
