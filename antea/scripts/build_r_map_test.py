import os

from hypothesis  import given, settings
import hypothesis.strategies as st

from . build_r_map import build_r_map

threshold = st.integers(min_value=0, max_value=10)

@settings(deadline=None)
@given(threshold)
def test_build_r_map(ANTEADATADIR, config_tmpdir, threshold):
    """
    Checks that the script to reconstruct the coincidences runs.
    """
    input_file  = os.path.join(ANTEADATADIR,  'full_body_coinc.h5')
    output_file = os.path.join(config_tmpdir, 'test_run_script')

    try:
        build_r_map(input_file, output_file, threshold)
    except:
        raise AssertionError('Function build_r_map has failed running.')
