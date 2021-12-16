import os

from . process_data_petit import process_data_petit


def test_process_data_petit(ANTEADATADIR, config_tmpdir):
    """
    Checks that the function to process petit data works.
    """
    input_file  = os.path.join(ANTEADATADIR,  'petit_data_test.h5')
    output_file = os.path.join(config_tmpdir, 'test_petit_data_process')

    try:
        process_data_petit(input_file, output_file)
    except:
        raise AssertionError('Function process_data_petit has failed running.')
