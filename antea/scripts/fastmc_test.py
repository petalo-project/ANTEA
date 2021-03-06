import os

from . fastmc import fastmc

def test_fastmc(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to create coincidences from fast simulation runs.
    """
    input_file  = os.path.join(ANTEADATADIR,  'full_body_fastsim_test.h5')
    output_file = os.path.join(config_tmpdir, 'test_run_script.h5')

    try:
        fastmc(input_file, output_file, ANTEADATADIR)
    except:
        raise AssertionError('Function fastmc has failed running.')
