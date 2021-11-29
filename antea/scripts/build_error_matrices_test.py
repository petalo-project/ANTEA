import os

from . build_error_matrices import build_error_matrices

def test_fastmc(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to build error matrices runs.
    """
    input_folder  = os.path.join(ANTEADATADIR, 'build_err_matrices_test/')

    try:
        build_error_matrices(input_folder, config_tmpdir, 'test')
    except:
        raise AssertionError('Function build_error_matrices has failed running.')
