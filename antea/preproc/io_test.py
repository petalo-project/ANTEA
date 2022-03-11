import pandas as pd
import tables as tb
import numpy  as np
import os

from glob import glob
from antea.preproc.io import compute_file_chunks_indices
from antea.preproc.io import write_corrected_df_daq
from antea.preproc.io import get_files


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


def test_write_corrected_df_daq(output_tmpdir):
    '''
    Check that the dataframe is written to file correctly
    '''
    filein = os.path.join(output_tmpdir, 'data_out.h5')
    df_1   = pd.DataFrame({'evt_number': [0, 0, 0, 0, 0],
                           'channel_id': [14, 29, 33, 47, 58],
                           'tac_id'    : [0, 3, 2, 1, 0]})

    df_2   = pd.DataFrame({'evt_number': [1, 1, 1, 1, 1],
                           'channel_id': [16, 25, 60, 37, 41],
                           'tac_id'    : [1, 3, 2, 0, 0]})

    write_corrected_df_daq(filein, df_1, 1, append = False)
    write_corrected_df_daq(filein, df_2, 2, append = True)

    df_out_1 = pd.read_hdf(filein, 'data_1')
    df_out_2 = pd.read_hdf(filein, 'data_2')

    np.testing.assert_array_equal(df_out_1, df_1)
    np.testing.assert_array_equal(df_out_2, df_2)


def test_get_files(output_tmpdir):
    '''
    Check that the correct files are found and they are returned sorted.
    '''
    run_number = 11200
    #Recreate data folder structure
    new_folder_pattern = os.path.join(output_tmpdir, 'analysis/{run}/hdf5/data/')
    new_folder = new_folder_pattern.format(run = run_number)
    os.makedirs(new_folder)

    #Create empty HDF5 files with file names unsorted
    sample_dir_1 = os.path.join(new_folder, 'data_0001.h5')
    sample_dir_2 = os.path.join(new_folder, 'data_0002.h5')
    sample_dir_3 = os.path.join(new_folder, 'data_0032.h5')
    sample_dir_4 = os.path.join(new_folder, 'data_0029.h5')

    df = pd.DataFrame()
    df.to_hdf(sample_dir_1, key = 'data')
    df.to_hdf(sample_dir_2, key = 'data')
    df.to_hdf(sample_dir_3, key = 'data')
    df.to_hdf(sample_dir_4, key = 'data')

    #Check if files are retrieved sorted by the file name
    files_expected = np.array([sample_dir_1, sample_dir_2, sample_dir_4, sample_dir_3])
    files          = get_files(run_number, folder = new_folder_pattern)

    np.testing.assert_array_equal(files, files_expected)
