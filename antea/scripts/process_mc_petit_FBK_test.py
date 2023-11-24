import os

from . process_mc_petit_FBK import process_mc_petit_FBK


def test_process_mc_petit_FBK(ANTEADATADIR, config_tmpdir):
    """
    Checks that the function to process petit data with FBK SiPMs works.
    """
    input_file  = os.path.join(ANTEADATADIR,  'petit_mc_fbk_test.pet.h5')
    output_file = os.path.join(config_tmpdir, 'test_petit_mc_fbk_process.h5')
    recovery_time = 80

    try:
        process_mc_petit_FBK(input_file, output_file, recovery_time)
    except:
        raise AssertionError('Function process_mc_petit_FBK has failed running.')
