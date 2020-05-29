import tables as tb
import numpy  as np

from . mc_io  import units_dict

from invisible_cities.core            import system_of_units as units
from invisible_cities.evm.event_model import Waveform

from typing import Mapping


def read_SiPM_bin_width_from_conf(h5f):

    h5config = h5f.root.MC.configuration
    bin_width = None
    for row in h5config:
        param_name = row['param_key'].decode('utf-8','ignore')
        if param_name.find('time_binning') >= 0:
            param_value = row['param_value'].decode('utf-8','ignore')
            numb, unit  = param_value.split()
            if param_name.find('SiPM') >= 0:
                bin_width = float(numb) * units_dict[unit]

    if bin_width is None:
        bin_width = 1 * units.microsecond

    return bin_width


def read_mcsns_response_evt (mctables: (tb.Table, tb.Table),
                             event_number: int, last_line_of_event,
                             bin_width, last_row=0) -> [tb.Table]:

    h5extents   = mctables[0]
    h5waveforms = mctables[1]

    current_event = {}
    event_range   = (last_row, int(1e9))

    iwvf = int(0)
    if event_range[0] > 0:
        iwvf = int(h5extents[event_range[0]-1][last_line_of_event]) + 1

    for iext in range(*event_range):
        this_row = h5extents[iext]
        if this_row['evt_number'] == event_number:
            # the index of the first waveform is 0 unless the first event
            #  written is to be skipped: in this case they must be read from the extents
            iwvf_end          = int(h5extents[iext][last_line_of_event])
            if iwvf_end < iwvf: break
            current_sensor_id = h5waveforms[iwvf]['sensor_id']
            time_bins         = []
            charges           = []
            while iwvf <= iwvf_end:
                wvf_row   = h5waveforms[iwvf]
                sensor_id = wvf_row['sensor_id']

                if sensor_id == current_sensor_id:
                    time_bins.append(wvf_row['time_bin'])
                    charges.  append(wvf_row['charge'])
                else:
                    times = np.array(time_bins) * bin_width
                    current_event[current_sensor_id] = Waveform(times, charges, bin_width)

                    time_bins = []
                    charges   = []
                    time_bins.append(wvf_row['time_bin'])
                    charges.append(wvf_row['charge'])

                    current_sensor_id = sensor_id

                iwvf += 1

            times     = np.array(time_bins) * bin_width
            current_event[current_sensor_id] = Waveform(times, charges, bin_width)
            break

    return current_event

def go_through_file(h5f, h5waveforms, event_range=(0, int(1e9)), bin_width=1.*units.microsecond, kind_of_waveform='data'):

    h5extents   = h5f.root.MC.extents
    sns_info    = (h5extents, h5waveforms)

    last_line_name = 'last_sns_' + kind_of_waveform
    events_in_file = len(h5extents)

    all_events     = {}
    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        evt_number = h5extents[iext]['evt_number']
        wvf_rows = read_mcsns_response_evt(sns_info, evt_number, last_line_name, bin_width, iext)
        all_events[evt_number] = wvf_rows

    return all_events

def read_mcsns_response(file_name, event_range=(0, int(1e9))) ->Mapping[int, Mapping[int, Waveform]]:

    kind_of_waveform = 'data'

    with tb.open_file(file_name, mode='r') as h5f:
        bin_width   = read_SiPM_bin_width_from_conf(h5f)
        h5waveforms = h5f.root.MC.waveforms
        all_events  = go_through_file(h5f, h5waveforms, event_range, bin_width, kind_of_waveform)

        return all_events

def read_mcTOFsns_response(file_name, event_range=(0, int(1e9))) ->Mapping[int, Mapping[int, Waveform]]:

    kind_of_waveform = 'tof'
    bin_width        = 5 * units.picosecond

    with tb.open_file(file_name, mode='r') as h5f:
        h5waveforms = h5f.root.MC.tof_waveforms
        all_events = go_through_file(h5f, h5waveforms, event_range, bin_width, kind_of_waveform)

        return all_events
