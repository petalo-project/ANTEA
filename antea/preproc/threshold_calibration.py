import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np

from matplotlib.dates import MINUTELY, SECONDLY
from matplotlib.dates import DateFormatter
from matplotlib.dates import rrulewrapper
from matplotlib.dates import RRuleLocator
from datetime         import datetime

from antea.preproc.io import get_files
from antea.preproc.io import get_evt_times


def filter_df_evts(df, evt_start, evt_end):
    '''
    Returns a dataframe filtered by a given range of the event numbers
    '''
    evt_filter = (df.evt_number >= evt_start) & (df.evt_number < evt_end)
    df_filtered = df[evt_filter]
    return df_filtered


def get_run_control(files):
    '''
    It returns a dataframe with data from all given files and it adds a new
    column of the run control differences. The run control is a bit that flips
    each time a start-stop occurs in the system configuration.
    '''
    dfs = []

    for i, fname in enumerate(files):
        df_tmp = pd.read_hdf(fname, 'dateEvents')
        df_tmp['fileno'] = i
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    df['diff'] = df.run_control.diff().fillna(0)
    return df


def compute_limit_evts_based_on_run_control(df_run):
    '''
    It returns a df with information about the event number and file number
    where the event starts and finishes.
    '''
    limits = df_run[df_run['diff'] != 0]

    start_evts = []
    end_evts   = []
    files1     = []
    files2     = []

    previous_start = 0
    file1 = 0
    file2 = 0

    for _, row in limits.iterrows():

        file1 = file2
        file2 = row.fileno
        start_evts.append(previous_start)
        end_evts  .append(row.evt_number)
        files1    .append(file1)
        files2    .append(file2)
        previous_start = row.evt_number


    start_evts.append(previous_start)
    end_evts  .append(df_run.evt_number.values[-1] + 1)
    files1    .append(row.fileno)
    files2    .append(df_run.fileno    .values[-1])

    # [start, end)
    df_limits = pd.DataFrame({'start' : start_evts,
                              'end'   : end_evts,
                              'file1' : files1,
                              'file2' : files2})
    return df_limits


def process_df(df, channels, field, value):
    '''
    It returns a dataframe with the statistical operations of count, mean, std,
    min, max and sum of the values selected for each vth_t1 or vth_t2. The field
    variable can be 'vth_t1' or 'vth_t2' and the value variable is the
    corresponding value of vth_t1 or vth_t2.
    '''
    df_filtered    = df[df.channel_id.isin(channels)][['channel_id', 'count']]
    operations     = ['count', 'mean', 'std', 'min', 'max', 'sum']
    df_agg         = df_filtered.groupby('channel_id').agg(operations)
    df_agg.columns = operations
    df_tmp         = df_agg.reset_index()
    df_tmp[field]  = value
    return df_tmp


def compute_max_counter_value_for_each_config(tofpet_id, field, channels, files, limits):
    '''
    It returns a dataframe with the std, mean, max, min and sum calculation
    per channel for each group of data and a tofpet events' array.
    '''
    results       = []
    tofpet_evts   = []
    current_file1 = -1
    df = pd.DataFrame()

    for iteration, limit in limits.iterrows():
        file1 = int(limit.file1)
        file2 = int(limit.file2)

        if file1 != current_file1:
            df            = pd.read_hdf(files[file1], 'counter')
            current_file1 = file1

        df_filtered = filter_df_evts(df, limit.start, limit.end)

        if file1 != file2:
            df2          = pd.read_hdf(files[file2], 'counter')
            df2_filtered = filter_df_evts(df2, limit.start, limit.end)
            df_filtered  = pd.concat([df_filtered, df2_filtered])

            # Update file1
            df            = df2
            current_file1 = file2

        df_filtered = df_filtered[df_filtered.tofpet_id == tofpet_id]

        tofpet_evts.append(df_filtered.shape[0])

        try:
            result = process_df(df_filtered, channels, field, iteration)
            results.append(result)
        except ValueError:
            print("Error")

    return pd.concat(results), tofpet_evts


def plot_evts_recorded_per_configuration(tofpet_evts, limits):
    '''
    It plots the tofpet events and date events
    '''
    date_evts = limits.start.diff().values[1:]
    max1 = 1.5 * np.max(tofpet_evts)
    max2 = 1.5 * np.max(date_evts)

    fig, ax1 = plt.subplots(figsize=(15,7))

    color = 'tab:red'
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('TOFPET evts', color=color)
    ax1.plot(tofpet_evts, color=color, drawstyle='steps')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, max1])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('DATE evts', color=color)  # we already handled the x-label with ax1
    ax2.plot(date_evts, color=color, drawstyle='steps', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max2])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_time_distribution(df_times):
    '''
    It plots the events time distribution.
    '''
    dtimes = [datetime.fromtimestamp(time) for time in (df_times.timestamp/1e6).values]

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(20,7))

    plt.plot_date(dtimes, np.ones_like(dtimes), '.')

    formatter = DateFormatter('%H:%M:%S')
    rule      = rrulewrapper(SECONDLY, interval=30)
    loc       = RRuleLocator(rule)

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30)
    ax.yaxis.set_visible(False)
    ax.set_title("Events time distribution")


def plot_channels(df_counts, channels, nbits):
    '''
    It plots the counts read in each channel vs the vth configured
    '''
    plt.rcParams.update({'font.size': 14})

    #Plot size for different number of channels available
    if len(channels) == 64:
        fig, ax = plt.subplots(figsize=(40,34))

    elif len(channels) == 48:
        fig, ax = plt.subplots(figsize=(40,26))

    elif len(channels) == 32:
        fig, ax = plt.subplots(figsize=(40,17))

    elif len(channels) == 16:
        fig, ax = plt.subplots(figsize=(40,8))

    rows = int(len(channels)/8)

    for i, ch in enumerate(channels):
        values = df_counts[df_counts.channel_id == ch]['max'].values
        ax     = plt.subplot(rows, 8, i+1)
        ymax   = (2**nbits)
        plt.plot(values, drawstyle='steps', linewidth=3)
        plt.ylim(0, 1.1*ymax)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.text(15, 0.5*ymax, f"ch: {ch}", horizontalalignment='center',
                 verticalalignment='center', rotation=0, fontsize=15)

        if i in [0, 1, 2, 8, 9, 10, 16, 17, 18]:
            plt.setp(ax.get_xticklabels(), fontsize=16)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)


def process_run(run, nbits, field, tofpet_id, channels, plot = False, folder = '/analysis/{run}/hdf5/data/'):
    '''
    It returns a df with the necessary arithmetic operations for each channel
    and vth value. Optionally it plots the time distribution and the events
    recorded per configuration.
    '''
    files                  = get_files(run, folder)
    df_times               = get_evt_times(files)
    df_run                 = get_run_control(files)
    limits                 = compute_limit_evts_based_on_run_control(df_run)
    df_counts, tofpet_evts = compute_max_counter_value_for_each_config(tofpet_id,
                             field, channels, files, limits)

    if plot:
        plot_time_distribution(df_times)
        plot_evts_recorded_per_configuration(tofpet_evts, limits)

    return df_counts


def plot_channels_multiple_runs(dfs, runs_dict, channels, title='', fname=None):
    '''
    It plots the channel counts vs the vth from different runs in the same
    subplot so that it is easier to compare them.
    '''
    runs = runs_dict['run_number']
    nbits = runs_dict['nbits']
    rows = int(len(channels)/8)

    #Plot size for different number of channels available
    if len(channels) == 64:
        fig, ax = plt.subplots(figsize=(40,34))

    elif len(channels) == 48:
        fig, ax = plt.subplots(figsize=(40,26))

    elif len(channels) == 32:
        fig, ax = plt.subplots(figsize=(40,17))

    elif len(channels) == 16:
        fig, ax = plt.subplots(figsize=(40,8))

    colors = iter(['red', 'blue', 'orange', 'green', 'brown', 'yellow'])

    for df_counts, run in zip(dfs, runs):
        color = next(colors)
        for i, ch in enumerate(channels):
            values = df_counts[df_counts.channel_id == ch]['max'].values
            ax     = plt.subplot(rows, 8, i+1)
            ymax   = (2**nbits)

            plt.plot(values, drawstyle='steps', linewidth=3, color=color,
                     alpha=0.5, label=run)
            plt.ylim(0, 1.1*ymax)
            plt.text(5, 0.5*ymax, f"ch: {ch}", horizontalalignment='center',
                     verticalalignment='center', rotation=0, fontsize=13)

            max_label = '2^{{{}}}'.format(nbits)
            ax.set_yticks([0, 2**nbits])
            ax.set_yticklabels(['$0$', f'${max_label}$'])
            plt.setp(ax.get_yticklabels(), fontsize=14)

            if i in [0, 1, 2, 8, 9, 10, 16, 17, 18]:
                plt.setp(ax.get_xticklabels(), fontsize=18)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(bbox_to_anchor=(1.5, rows + 1.3))
    plt.suptitle(title)
    fig.tight_layout()

    if fname:
        plt.savefig(fname)


def find_threshold(df_counter, nbits, vth_t, threshold_2 = None):
    #vth is a string, it can be vth_t1 or vth_t2
    '''
    It returns an array with the thresholds we are looking for. The function
    changes taking into account if we are looking for the vth_t1 thresholds
    (cutting noise in 0) or the vth_t2 thresholds.
    '''
    vth_1    = np.zeros(64)
    channels = df_counter['channel_id'].unique()
    channels.sort()

    if   vth_t == 'vth_t1':
        valor = 'max'
    elif vth_t == 'vth_t2':
        valor = 'expected_rate'

    for i, ch in enumerate(channels):
        df_filtered = df_counter[df_counter['channel_id']==ch]
        waveform    = df_filtered[valor].values # max for vth_t1 or expected_rate for vth_t2
        vth_t1s     = df_filtered[vth_t].values

        if vth_t == 'vth_t1':
            threshold_v1 = 1

            for j, (count, vth_t1) in enumerate(zip(waveform, vth_t1s)):
                #We look for the first data that is no null and we save its
                #vth_t1. If higher than 10, we save the previous one
                if threshold_v1 <= count:
                    vth_1[i] = vth_t1

                    if count > 10:
                        vth_1[i] = vth_t1 - 1

                    break

        elif vth_t == 'vth_t2':

            threshold_v2 = threshold_2 # Expected rate where we want to cut the
                                       # noise taking into account the activity fit

            for j, (count, vth_t1) in enumerate(zip(waveform, vth_t1s)):
                #we look for the last data that is null from right to left

                if count > threshold_v2:
                    vth_1[i] = vth_t1

                    break

    return vth_1
