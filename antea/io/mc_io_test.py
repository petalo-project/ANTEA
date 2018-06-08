import os
import numpy  as np
import tables as tb

from . mc_io import read_mcsns_response


def test_read_sensor_response():
    test_file = os.environ['ANTEADIR'] + '/testdata/full_ring_test.pet.h5'

    mc_sensor_dict = read_mcsns_response(test_file)
    waveforms = mc_sensor_dict[0]

    n_of_sensors = 796
    sensor_id    = 4162
    
    assert len(waveforms) == n_of_sensors
    assert waveforms[sensor_id].times == np.array([0.])
    assert waveforms[sensor_id].charges == np.array([8.])

def test_read_last_sensor_response():
    test_file = os.environ['ANTEADIR'] + '/testdata/full_ring_test.pet.h5'

    mc_sensor_dict = read_mcsns_response(test_file)
    waveforms = mc_sensor_dict[0]

    with tb.open_file(test_file, mode='r') as h5in:
        last_written_id = h5in.root.MC.sensor_positions[-1][0]
        last_read_id = list(waveforms.keys())[-1]

        assert last_read_id == last_written_id
