import tables as tb
import pandas as pd

from invisible_cities.core import system_of_units as units

from typing import Mapping

str_length = 20
units_dict = {'picosecond' : units.picosecond,  'ps' : units.picosecond,
              'nanosecond' : units.nanosecond,  'ns' : units.nanosecond,
              'microsecond': units.microsecond, 'mus': units.microsecond,
              'millisecond': units.millisecond, 'ms' : units.millisecond}

class mc_sns_response_writer:
    """Add MC sensor response info to existing file."""
    def __init__(self, filename: str, sns_df_name: str = 'waveforms_lut'):

        self.filename = filename
        self.sns_df_name = sns_df_name

        with tb.open_file(filename, 'r+') as h5in:
            if self.sns_df_name in h5in.root.MC:
                h5in.remove_node('/MC/'+self.sns_df_name, recursive=True)

        self.store = pd.HDFStore(filename, "a", complib=str("zlib"), complevel=4)


    def close_file(self):
        self.store.close()


    def __call__(self, sns_response: Mapping[int, Mapping[int, float]], evt_number: int):

        waveforms_dict = sns_response[evt_number]
        waveforms = pd.DataFrame({'event_id':  [evt_number for i in range(len(waveforms_dict))],
                                  'sensor_id': list(waveforms_dict.keys()),
                                  'time_bin':  [0 for i in range(len(waveforms_dict))],
                                  'charge':    list(waveforms_dict.values())})
        self.store.append('MC/'+self.sns_df_name, waveforms, format='t', data_columns=True)


class mc_writer:
    """Copy MC true info to output file."""
    def __init__(self, filename_in: str, filename_out: str):

        self.store = pd.HDFStore(filename_out, "a", complib=str("zlib"), complevel=4)
        conf = load_configuration(filename_in)
        self.store.append('MC/configuration', conf, format='t', data_columns=True)

        self.hits      = load_mchits(filename_in)
        self.particles = load_mcparticles(filename_in)


    def close_file(self):
        self.store.close()


    def __call__(self, evt_number: int):

        evt_hits      = self.hits[self.hits.event_id == evt_number]
        evt_particles = self.particles[self.particles.event_id == evt_number]
        self.store.append('MC/hits',      evt_hits,      format='t', data_columns=True,
                          min_itemsize={'label' : str_length})
        self.store.append('MC/particles', evt_particles, format='t', data_columns=True,
                          min_itemsize={'name' : str_length, 'initial_volume' : str_length,
                                        'final_volume' : str_length, 'creator_proc': str_length})


def read_sensor_bin_width_from_conf(h5f, tof=False):
    """
    Return the time bin width (either TOF or no TOF) with units.
    """

    h5config = h5f.root.MC.configuration
    bin_width = None
    binning = 'bin_size'
    if tof:
        binning = 'tof_bin_size'
    for row in h5config:
        param_name = row['param_key'].decode('utf-8','ignore')
        if param_name.find(binning) >= 0:
            param_value = row['param_value'].decode('utf-8','ignore')
            numb, unit  = param_value.split()
            bin_width = float(numb) * units_dict[unit]

    return bin_width


def load_mchits(file_name: str) -> pd.DataFrame:

    hits = pd.read_hdf(file_name, 'MC/hits')

    return hits


def load_mcparticles(file_name: str) -> pd.DataFrame:

    particles = pd.read_hdf(file_name, 'MC/particles')

    return particles


def load_mcsns_response(file_name: str) -> pd.DataFrame:

    sns_response = pd.read_hdf(file_name, 'MC/waveforms')

    return sns_response


def load_mcTOFsns_response(file_name: str) -> pd.DataFrame:

    sns_response = pd.read_hdf(file_name, 'MC/tof_waveforms')

    return sns_response


def load_configuration(file_name: str) -> pd.DataFrame:

    conf = pd.read_hdf(file_name, 'MC/configuration')

    return conf
