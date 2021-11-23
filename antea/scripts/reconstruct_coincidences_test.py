import os

from .. database import load_db as db
from . reconstruct_coincidences import reconstruct_coincidences_script


def test_characterize_coincidences(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to reconstruct the coincidences runs.
    """
    input_file  = os.path.join(ANTEADATADIR, 'full_body_coinc.h5')
    output_file = os.path.join(config_tmpdir, 'test_run_script')
    rpos_file   = os.path.join(ANTEADATADIR, 'r_table_full_body.h5')

    DataSiPM = db.DataSiPMsim_only('petalo', 0)

    try:
        characterize_coincidences(input_file, output_file, rpos_file, DataSiPM)
    except:
        raise AssertionError('Function reconstruct_coincidences_script has failed running.')


