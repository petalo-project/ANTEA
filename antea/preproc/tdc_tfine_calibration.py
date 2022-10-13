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
