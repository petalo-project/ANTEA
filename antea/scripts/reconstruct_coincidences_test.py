import os

from contextlib import contextmanager

from .. database import load_db as db
from . reconstruct_coincidences import reconstruct_coincidences_script

@contextmanager
def does_not_raise():
    yield

def test_run_script(ANTEADATADIR, config_tmpdir):
    """
    Checks that the script to reconstruct the coincidences runs.
    """
    input_file  = os.path.join(ANTEADATADIR, 'ring_test_1000ev.h5')
    output_file = os.path.join(config_tmpdir, 'electrons_40keV_z250_RWF.h5')
    rpos_file   = os.path.join(ANTEADATADIR, 'r_table_full_body.h5')

    DataSiPM = db.DataSiPMsim_only('petalo', 0)

    with does_not_raise():
        reconstruct_coincidences_script(input_file, output_file, rpos_file, DataSiPM)

