import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np

from matplotlib.dates import MINUTELY, SECONDLY
from matplotlib.dates import DateFormatter
from matplotlib.dates import rrulewrapper
from matplotlib.dates import RRuleLocator

from glob             import glob
from datetime         import datetime
from sklearn.cluster  import KMeans
from scipy.special    import erf

from antea.preproc.io                    import get_files
from antea.preproc.threshold_calibration import get_run_control
from antea.preproc.threshold_calibration import compute_limit_evts_based_on_run_control
from antea.preproc.threshold_calibration import filter_df_evts
from antea.preproc.threshold_calibration import plot_evts_recorded_per_configuration

from invisible_cities.core.fit_functions import fit
from invisible_cities.core.fit_functions import gauss


def process_df_to_assign_tpulse_delays(files, limits, configs):
    '''
    It returns a dataframe adding a column with the delay configuration
    for each row in limits and a tofpet events' array. In addition, it also
    returns the number of wrong rows in the Dataframe and which channel_id are.
    '''
    results       = []
    tofpet_evts   = []
    current_file1 = -1
    df = pd.DataFrame()

    wrong_rows = 0
    wrong_ch_t = np.array([])

    for iteration, limit in limits.iterrows():
        file1 = int(limit.file1)
        file2 = int(limit.file2)

        if file1 != current_file1:
            df            = pd.read_hdf(files[file1], 'data')
            current_file1 = file1

        df_filtered = filter_df_evts(df, limit.start, limit.end)

        if file1 != file2:
            df2          = pd.read_hdf(files[file2], 'data')
            df2_filtered = filter_df_evts(df2, limit.start, limit.end)
            df_filtered  = pd.concat([df_filtered, df2_filtered])

            # Update file1
            df            = df2
            current_file1 = file2

        df_filt = df_filtered.groupby('channel_id').count()['tfine'].reset_index()

        ch_max        = df_filt.loc[df_filt.loc[:, 'tfine'] == df_filt['tfine'].max()].channel_id.sum()
        wrong_chs     = df_filt.channel_id.values
        wrong_ch_del  = wrong_chs[wrong_chs != ch_max]
        wrong_ch_t    = np.append(wrong_ch_t, wrong_ch_del)
        wrong_rows    = wrong_rows + df_filt.loc[df_filt.loc[:, 'tfine'] != df_filt['tfine'].max()]['tfine'].sum()

        df_filtered   = df_filtered[df_filtered['channel_id'] == ch_max]

        tofpet_evts.append(df_filtered.shape[0])

        df_filtered['delay'] = configs[iteration]/1000
        results.append(df_filtered)

    return pd.concat(results), tofpet_evts, wrong_rows, wrong_ch_t


def compute_normalized_histogram(values, hist_range, bins):
    '''
    It returns the counts and edges to plot a normalized histogram given
    the values array, range and bins wanted.
    '''
    counts, xedges = np.histogram(values, range = hist_range, bins = bins)

    xstep  = xedges[1]   - xedges[0]
    xs     = xedges[:-1] + xstep/2
    counts = counts / counts.sum()

    return counts, xs


def plot_phase_histograms(df, hist_range, bins, phases, text=True, offset=0):
    '''
    It plots some channel phases distribution to see its tfine range.
    '''
    fig, ax = plt.subplots(figsize=(10,7))
    plt.rc('font', size=12)
    ax      = plt.subplot(1, 1, 1)

    for phase in phases:
        tfines     = df[df.delay == phase].tfine.values
        counts, xs = compute_normalized_histogram(tfines, hist_range, bins)

        plt.hist(xs, weights=counts, range=hist_range, bins=bins, alpha=0.5,
                 label="Phase {}".format(phase))
        if text:
            if phase in [330,270,210,150,90,30]:
                plt.text(counts.argmax() + offset, 0.04, f"{phase}",
                         horizontalalignment='center', verticalalignment='center',
                         rotation=90, fontsize=14)
            elif phase in [360,300,240,180,120,60,0]:
                plt.text(counts.argmax() + offset, 0.03, f"{phase}",
                         horizontalalignment='center', verticalalignment='center',
                         rotation=90, fontsize=14)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels,bbox_to_anchor=(1.15, 0.5),loc = 'center right')


def fit_gaussian(values, plot=False, text = False):
    '''
    It returns the gaussian fit parameters, their errors
    and the fit chi squared.
    '''
    hist_range = [values.min(), values.max()]
    bins       = np.int(hist_range[1] - hist_range[0])

    counts, xedges = np.histogram(values, range=hist_range, bins=bins)
    xstep          = xedges[1] - xedges[0]
    xs             = xedges[:-1] + xstep/2

    fit_result               = fit(gauss, xs, counts, (100., np.mean(values),
                                   np.sqrt(np.var(values))))
    amp, mu, sigma           = fit_result.values
    err_amp,err_mu,err_sigma = fit_result.errors
    chi2                     = fit_result.chi2

    if plot:
        plt.plot(xs, fit_result.fn(xs), linewidth=2,label='fit')
        plt.hist(xs, weights=counts, range=hist_range, bins=bins,label = 'counts');
        plt.legend()

    if text:
        print('The center of the gaussian fit is', mu)
        print('The error of the gaussian center is',err_mu)
        print('The sigma of the gaussian fit is', sigma)
        print('The error of the gaussian sigma is', err_sigma)
        print('The chi2 is', chi2)

    return mu, err_mu, sigma, err_sigma, chi2