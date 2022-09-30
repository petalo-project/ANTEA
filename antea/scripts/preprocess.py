from antea.preproc.tdc_corrections import correct_tfine_wrap_around
from antea.preproc.qdc_corrections import correct_efine_wrap_around

from antea.preproc.tdc_corrections import apply_tdc_correction
from antea.preproc.qdc_corrections import compute_efine_correction_using_linear_interpolation
from antea.preproc.qdc_corrections import create_qdc_interpolator_df

from antea.preproc.tdc_corrections import compute_integration_window_size
from antea.preproc.tdc_corrections import add_tcoarse_extended_to_df

from antea.preproc.clustering      import compute_evt_number_combined_with_cluster_id
from antea.preproc.io              import compute_file_chunks_indices
from antea.preproc.io              import write_corrected_df_daq

import pandas as pd


# Calibtration files
qdc0_fname = '/home/nsalor/petalo_calibration_nb/QDC_CALIBRATION/qdc_calibration_cold_compressor_asic584_tofpet5_imirror25_r12329.h5'
qdc2_fname = '/home/nsalor/petalo_calibration_nb/QDC_CALIBRATION/qdc_calibration_cold_compressor_asic581_tofpet1_imirror25_r12320.h5'

tdc0_fname = '/home/nsalor/petalo_calibration_nb/TDC_CALIBRATION/calibration_tdc_tfine_fine_cold_compressor_TOFPET5_run12332_sawtooth_fit.h5'
tdc2_fname = '/home/nsalor/petalo_calibration_nb/TDC_CALIBRATION/calibration_tdc_tfine_fine_cold_compressor_TOFPET1_run12322_sawtooth_fit.h5'


# Read calibration files
df_qdc       = create_qdc_interpolator_df(qdc0_fname, qdc2_fname)
df_tdc_asic0 = pd.read_hdf(tdc0_fname, key='tfine')
df_tdc_asic2 = pd.read_hdf(tdc2_fname, key='tfine')


# Set input and output files
filein  = '/analysis/12334/hdf5/data/run_12334_0000_trigger1_waveforms.h5'
fileout = 'test_file.h5'


def process_daq_df(df, df_qdc, df_tdc_asic0, df_tdc_asic2):
    compute_integration_window_size(df)

    correct_tfine_wrap_around(df)
    correct_efine_wrap_around(df)

    # Uncomment to remove events with tcoarse higher than a threshold
    # df.drop(df[df.tcoarse > 60000].index, inplace=True)

    add_tcoarse_extended_to_df(df)

    df_0 = df[df.tofpet_id == 5]
    df_2 = df[df.tofpet_id == 1]

    df_0 = apply_tdc_correction(df_0, df_tdc_asic0)
    df_2 = apply_tdc_correction(df_2, df_tdc_asic2)

    df = pd.concat([df_0, df_2]).sort_index()

    compute_efine_correction_using_linear_interpolation(df, df_qdc)

    df.drop(columns=['card_id', 'wordtype_id'], inplace=True)
    compute_evt_number_combined_with_cluster_id(df)
    return df


def process_daq_file(filein, fileout, df_qdc, df_tdc_asic0, df_tdc_asic2):
    chunks = compute_file_chunks_indices(filein)
    nchunks = chunks.shape[0]

    for i in range(nchunks-1):
        print("{}/{}".format(i, nchunks-2))
        start = chunks[i]
        end   = chunks[i+1]

        df = pd.read_hdf(filein, 'data', start=start, stop=end+1)
        df_corrected = process_daq_df(df, df_qdc, df_tdc_asic0, df_tdc_asic2)
        write_corrected_df_daq(fileout, df_corrected, i, i>0)


process_daq_file(filein, fileout, df_qdc, df_tdc_asic0, df_tdc_asic2)
