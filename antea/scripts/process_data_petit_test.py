import os

from . process_data_petit import process_data_petit


def test_process_data_petit(ANTEADATADIR, config_tmpdir):
    """
    Checks that the function to extract data from data runs works.
    """
    input_file  = os.path.join(ANTEADATADIR,  'data_petbox_test.h5')
    output_file = os.path.join(config_tmpdir, 'test_data_extract_charge')

    try:
        process_data_petit(input_file, output_file)
    except:
        raise AssertionError('Function process_data_petit has failed running.')
