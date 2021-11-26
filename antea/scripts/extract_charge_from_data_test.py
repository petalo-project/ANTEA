import os

from . extract_charge_from_data import extract_charge_from_data


def test_extract_charge_from_data(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to extract data from data runs.
    """
    input_file  = os.path.join(ANTEADATADIR,  'data_petbox_test.h5')
    output_file = os.path.join(config_tmpdir, 'test_data_extract_charge')

    try:
        extract_charge_from_data(input_file, output_file)
    except:
        raise AssertionError('Function extract_charge_from_data has failed running.')
