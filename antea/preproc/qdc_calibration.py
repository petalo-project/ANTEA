import matplotlib.pylab as plt

from antea.preproc.io import get_files
from antea.preproc.io import read_run_data
from antea.preproc.qdc_corrections import compute_qdc_calibration_using_mode


def plot_all_channels(df, xlim=None):
    '''
    Plots the efine vs the integration window size for each channel and tac id 0
    '''
    for ch in df.channel_id.unique():
        df_filtered = df[(df.channel_id == ch) & (df.tac_id == 0)]
        plt.plot(df_filtered.intg_w, df_filtered.efine)

    if xlim:
        plt.xlim(*xlim)


def process_qdc_calibration_run(run_number, max_intg_w = 291, hist=True,
                                folder = '/analysis/{run}/hdf5/data/'):
    '''
    It returns a df with the integration window size and the corrected efine.
    Optionally it plots a channel id and an integration window histogram.
    '''
    files   = get_files(run_number, folder)
    df_data = read_run_data(files)
    df      = compute_qdc_calibration_using_mode(df_data, max_intg_w)
    if hist:
        plt.figure()
        df_data.channel_id.hist(bins=64,  range=[0, 64])
        plt.figure()
        df_data.intg_w    .hist(bins=100, range=[0, 220])
    return df
