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
from antea.preproc.fit_functions         import skewnormal_function

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


def fit_semigaussian(values, plot=False, text = False):
    '''
    It returns the skewnormal fit parameters, their errors
    and the fit chi squared.
    '''
    hist_range = [values.min(), values.max()]
    bins       = hist_range[1] - hist_range[0]

    counts, xedges = np.histogram(values, range = hist_range, bins = bins)
    xstep          = xedges[1]   - xedges[0]
    xs             = xedges[:-1] + xstep/2

    fit_result     = fit(skewnormal_function, xs, counts, (0.5, np.median(values),
                         np.sqrt(np.median(values)),1000))

    chi2                                         = fit_result.chi2
    shape, location, scale, gain                 = fit_result.values
    err_shape, err_location, err_scale, err_gain = fit_result.errors

    #Convert skewnormal parameters to gaussian parameters:

    d         = shape / np.sqrt(1 + shape**2)
    err_d     = err_shape / (1 + shape**2)**(3/2)

    comun     = np.sqrt(2 / np.pi) * d - (4 - np.pi)/4 * ((d*np.sqrt(2/np.pi))**3 /
                (1 - 2 * d**2 / np.pi)**(3/2) * np.sqrt(1 - 2 * d**2 /
                np.pi)) - np.sign(shape) * np.exp(-2*np.pi / np.abs(shape))/2
    err_comun = np.sqrt((err_shape * np.pi * np.exp(-2*np.pi/np.abs(shape)) / shape**2)**2 +
                        (err_d * (-2*(np.pi - 8)*d**4 + np.pi*(3*np.pi - 20)*d**2 +
                        2*np.pi**2) / (np.sqrt(2*np.pi) * (np.pi - 2*d**2)**2))**2)

    moda      = location + scale * comun
    err_moda  = np.sqrt(err_location**2 + (err_scale*comun)**2 + (scale*err_comun)**2)

    sigma     = np.abs(scale * np.sqrt(1 - (2*d**2/np.pi)))
    err_sigma = np.sqrt((err_scale * np.sqrt(1 - (2*d**2 / np.pi)))**2 +
                        (err_d * 2 * scale * d / (np.sqrt(np.pi) *
                        np.sqrt(np.pi - 2*d**2)))**2)

    if plot:
        plt.plot(xs, fit_result.fn(xs), linewidth=2,label='fit')
        plt.hist(xs, weights=counts, range=hist_range, bins=bins,label = 'counts');
        plt.legend()

    if text:
        print('The center of the semigaussian is ',      moda)
        print('The error of the semigaussian center is ',err_moda)
        print('The sigma of the semigaussian is ',       sigma)
        print('The error of the semigaussian sigma is ', err_sigma)
        print('The chi2 is ',                            chi2)
        print('The shape is ',                           shape)

    return moda, err_moda, sigma, err_sigma, chi2, shape, location, scale


def one_distribution_fit(data_fit):
    '''
    It applies the semigaussian fit to the distribution unless it can't fit it,
    and then it fits a gaussian function.
    '''
    try:

        mode, err_mode, sigma, err_sigma, chi, shape, location, scale = fit_semigaussian(data_fit)

        mode_r      = mode_l      = mode
        sigma_r     = sigma_l     = sigma
        chi_r       = chi_l       = chi
        err_mode_r  = err_mode_l  = err_mode
        err_sigma_r = err_sigma_l = err_sigma

    except(RuntimeError):

        mode, err_mode, sigma, err_sigma, chi = fit_gaussian(data_fit)

        mode_r      = mode_l      = mode
        sigma_r     = sigma_l     = sigma
        chi_r       = chi_l       = chi
        err_mode_r  = err_mode_l  = err_mode
        err_sigma_r = err_sigma_l = err_sigma

    return [[mode_l, np.abs(sigma_l), mode_r, np.abs(sigma_r)],
            [err_mode_l, err_sigma_l, err_mode_r, err_sigma_r], [chi_l, chi_r]]


def two_distributions_fit(data_fit, percentage):
    '''
    It applies the semigaussian fit to both distributions. If one
    distribution has got more data than the parameter "percentage"
    it does the fit to the bigger distribution. If not, it does both fits
    '''
    range_fit = data_fit.max() - data_fit.min()

     # Fit Left
    data_fit_sel  = data_fit[data_fit < (range_fit/2.0 + data_fit.min())]
    range_fit_sel = data_fit_sel.max() - data_fit_sel.min()

    div = len(data_fit_sel)/len(data_fit)

    if (range_fit_sel == 0): #Only one bin left
        mode_l, sigma_l, chi_l = [data_fit_sel.min(), 1, 0]

    elif (div > (percentage/100)): #Almost all data is on the left side

        try: # Left big semigaussian
            fit_result  = fit_semigaussian(data_fit_sel)

            mode_r      = mode_l      = fit_result[0]
            err_mode_r  = err_mode_l  = fit_result[1]
            sigma_r     = sigma_l     = fit_result[2]
            err_sigma_r = err_sigma_l = fit_result[3]
            chi_r       = chi_l       = fit_result[4]

        except(RuntimeError): #Left big gaussian
            fit_result  = fit_gaussian(data_fit_sel)

            mode_r      = mode_l      = fit_result[0]
            err_mode_r  = err_mode_l  = fit_result[1]
            sigma_r     = sigma_l     = fit_result[2]
            err_sigma_r = err_sigma_l = fit_result[3]
            chi_r       = chi_l       = fit_result[4]

    elif ((div > (100-percentage)/100) & (div < (percentage/100))):

        try: #Left equal semigaussian
            fit_result  = fit_semigaussian(data_fit_sel)

            mode_l      = fit_result[0]
            err_mode_l  = fit_result[1]
            sigma_l     = fit_result[2]
            err_sigma_l = fit_result[3]
            chi_l       = fit_result[4]

        except(RuntimeError): #Left equal gaussian
            fit_result  = fit_gaussian(data_fit_sel)

            mode_l      = fit_result[0]
            err_mode_l  = fit_result[1]
            sigma_l     = fit_result[2]
            err_sigma_l = fit_result[3]
            chi_l       = fit_result[4]


    #Fit Right
    data_fit_sel  = data_fit[data_fit > (range_fit/2.0 + data_fit.min())]
    range_fit_sel = data_fit_sel.max() - data_fit_sel.min()

    div = len(data_fit_sel)/len(data_fit)

    if (range_fit_sel == 0): #Only one bin left
        mode_r, sigma_r, chi_r = [data_fit_sel.min(), 1, 0]

    elif (div > (percentage/100)): #Almost all data is on the right side

        try: # Right big semigaussian
            fit_result  = fit_semigaussian(data_fit_sel)

            mode_l      = mode_r      = fit_result[0]
            err_mode_l  = err_mode_r  = fit_result[1]
            sigma_l     = sigma_r     = fit_result[2]
            err_sigma_l = err_sigma_r = fit_result[3]
            chi_l       = chi_r       = fit_result[4]


        except(RuntimeError): # Right big gaussian
            fit_result  = fit_gaussian(data_fit_sel)

            mode_l      = mode_r      = fit_result[0]
            err_mode_l  = err_mode_r  = fit_result[1]
            sigma_l     = sigma_r     = fit_result[2]
            err_sigma_l = err_sigma_r = fit_result[3]
            chi_l       = chi_r       = fit_result[4]

    elif ((div > (100-percentage)/100) & (div < (percentage/100))):

        try: # Right equal semigaussian
            fit_result  = fit_semigaussian(data_fit_sel)

            mode_r      = fit_result[0]
            err_mode_r  = fit_result[1]
            sigma_r     = fit_result[2]
            err_sigma_r = fit_result[3]
            chi_r       = fit_result[4]

        except(RuntimeError): # Right equal gaussian
            fit_result  = fit_gaussian(data_fit_sel)

            mode_r      = fit_result[0]
            err_mode_r  = fit_result[1]
            sigma_r     = fit_result[2]
            err_sigma_r = fit_result[3]
            chi_r       = fit_result[4]

    return [[mode_l, np.abs(sigma_l), mode_r, np.abs(sigma_r)],
           [err_mode_l, err_sigma_l, err_mode_r, err_sigma_r], [chi_l, chi_r]]

def select_fitting_distribution(data_fit, percentage):
    '''
    It selects the two distributions fit or the one distribution
    fit depending on the data standard deviation.
    '''

    if  data_fit.std() > 10: # Two distributions
        try:
            fit_values = two_distributions_fit(data_fit, percentage)

        except:
            raise RuntimeError('Error in fitting two distributions')

    else:                    # One distribution
        try:
            fit_values  = one_distribution_fit(data_fit)

        except:
            raise RuntimeError('Error in fitting one distribution')

    return fit_values

def fit_all_channel_phases(filename, channels, percentage):
    '''
    It fits all channel's phases and return the results in a dataframe.
    '''

    res = []

    for ch in channels:
        data_ch = pd.read_hdf(filename,key= 'ch' + str(ch))
        tac = data_ch['tac_id'].unique()

        for tc in tac:
            data_tc = data_ch[data_ch['tac_id']==tc]

            # Find delay arrays:
            delays = data_tc['delay'].unique()

            for i in delays:
                data_fit = data_tc[data_tc['delay']==i].tfine

                if data_fit.size > 0:

                    fit_values = select_fitting_distribution(data_fit, percentage)

                    res.append([ch, tc, i, fit_values[0][0], fit_values[1][0],
                                fit_values[0][1], fit_values[1][1], fit_values[2][0],
                                fit_values[0][2], fit_values[1][2], fit_values[0][3],
                                fit_values[1][3], fit_values[2][1]])

    df_tfine_semigaus = pd.DataFrame(res,columns=['channel_id','tac_id','phase',
                                         'mode_l','err_mode_l','sigma_l','err_sigma_l',
                                         'chi_l', 'mode_r','err_mode_r', 'sigma_r',
                                         'err_sigma_r', 'chi_r'])
    return df_tfine_semigaus
