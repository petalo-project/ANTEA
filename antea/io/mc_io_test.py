import os
import numpy  as np
import tables as tb

from invisible_cities.core import system_of_units as units

from . mc_io import load_mcsns_response
from . mc_io import load_mcTOFsns_response


def test_read_sensor_response():
    test_file = os.environ['ANTEADIR'] + '/testdata/ring_test.h5'

    waveforms = load_mcsns_response(test_file)

    evt_id       = 16
    n_of_sensors = 90
    sensor_id, charge = 2945, 3

    evt_waveforms = waveforms[waveforms.event_id == evt_id]
    sns_charge    = evt_waveforms[evt_waveforms.sensor_id == sensor_id].charge.sum()

    assert len(evt_waveforms.sensor_id.unique()) == n_of_sensors
    assert sns_charge == charge

def test_read_sensor_tof_response():
    test_file = os.environ['ANTEADIR'] + '/testdata/ring_test.h5'

    waveforms = load_mcTOFsns_response(test_file)

    evt_id    = 16
    sensor_id = -2945
    times     = np.array([200, 1025, 3271])
    charges   = np.array([1, 1, 1])

    evt_waveforms = waveforms[waveforms.event_id == evt_id]
    evt_sns_waveforms = evt_waveforms[evt_waveforms.sensor_id == sensor_id]

    assert np.allclose(evt_sns_waveforms.time_bin, times)
    assert np.allclose(evt_sns_waveforms.charge, charges)
