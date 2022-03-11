import pandas as pd
import tables as tb
import numpy  as np
import os

from glob import glob
from antea.preproc.io import compute_file_chunks_indices


def test_compute_chunks_indices(output_tmpdir):
    '''
    Check that the data split takes into account the event number
    '''
    filein = os.path.join(output_tmpdir, 'data.h5')

    df = pd.DataFrame({'evt_number': [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3]})

    #Create hdf5 test file with the proper structure
    df.to_hdf(filein, key = 'dummy', format='table', data_columns = True)
    with tb.open_file(filein, 'a') as h5in:
        h5in.move_node  (h5in.root.dummy.table, newparent=h5in.root, newname='data')
        h5in.remove_node(h5in.root.dummy)


    chunks          = compute_file_chunks_indices(filein, chunk_size = 5)
    expected_chunks = np.array([0, 4, 8, 10, 15])

    np.testing.assert_array_equal(chunks, expected_chunks)
