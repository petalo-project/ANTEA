import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pylab as plt
import os

from antea.preproc.io import get_files
from antea.preproc.io import read_run_data
from antea.preproc.qdc_corrections import compute_qdc_calibration_using_mode
from antea.preproc.qdc_calibration import process_qdc_calibration_run


def test_process_qdc_calibration_run(output_tmpdir):
    '''
    Check that the dataframe is returned with the integration window and the
    corrected efine.
    '''
    run_number = 11229
    #Recreate data folder structure
    new_folder_pattern = os.path.join(output_tmpdir, 'analysis/{run}/hdf5/data/')
    new_folder         = new_folder_pattern.format(run = run_number)
    os.makedirs(new_folder)

    #Create HDF5 files with file names unsorted
    sample_dir_11 = os.path.join(new_folder, 'data_0011.h5')
    sample_dir_6  = os.path.join(new_folder, 'data_0006.h5')

    first_df  = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0],
                              'channel_id': [15, 15, 14, 17, 14],
                              'tac_id'    : [1, 2, 3, 0, 1],
                              'efine'     : [350, 225, 290, 360, 150],
                              'ecoarse'   : [1024, 300, 250, 800, 525],
                              'tcoarse'   : [59000, 200, 1024,  3600, 40200]})

    second_df = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0],
                              'channel_id': [14, 17, 15, 17, 14],
                              'tac_id'    : [0, 1, 2, 3, 0],
                              'efine'     : [50, 100, 125, 300, 200],
                              'ecoarse'   : [150, 582, 6, 700, 1000],
                              'tcoarse'   : [100, 4387, 28675, 33400, 40800]})

    first_df. to_hdf(sample_dir_11, key = 'data', format = 'table')
    second_df.to_hdf(sample_dir_6,  key = 'data', format = 'table')

    #Check the function output
    df_calib = process_qdc_calibration_run(run_number, hist=False, folder=new_folder_pattern)
    df_expected = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'channel_id': [14, 14, 14, 14, 15, 15, 15, 17, 17, 17],
                                'tac_id'    : [0, 0, 1, 3, 1, 2, 2, 0, 1, 3],
                                'intg_w'    : [50, 136, 261, 250, 392, 0, 100, 272, 2000, 68],
                                'efine'     : [64, 214, 164, 304, 364, 0, 239, 374, 114, 314]})

    np.testing.assert_array_equal(df_calib, df_expected)
