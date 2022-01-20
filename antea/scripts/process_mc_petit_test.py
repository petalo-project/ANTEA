import os

from . process_mc_petit import process_mc_petit


def test_process_mc_petit(ANTEADATADIR, config_tmpdir):
    """
    Checks that the function to process petit data works.
    """
    input_file  = os.path.join(ANTEADATADIR,  'petit_mc_test.pet.h5')
    output_file = os.path.join(config_tmpdir, 'test_petit_mc_process.h5')

    try:
        process_mc_petit(input_file, output_file)
    except:
        raise AssertionError('Function process_mc_petit has failed running.')
