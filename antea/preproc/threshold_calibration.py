import matplotlib.pylab as plt
import pandas           as pd
import tables           as tb
import numpy            as np

from matplotlib.dates import MINUTELY, SECONDLY
from matplotlib.dates import DateFormatter
from matplotlib.dates import rrulewrapper
from matplotlib.dates import RRuleLocator
from datetime         import datetime



def filter_df_evts(df, evt_start, evt_end):
    '''
    Returns a dataframe filtered by a given range of the event numbers
    '''
    evt_filter = (df.evt_number >= evt_start) & (df.evt_number < evt_end)
    df_filtered = df[evt_filter]
    return df_filtered


def get_run_control(files):
    '''
    It returns a dataframe with data from all files given and it adds a new
    column of the run control differences.
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


def process_df(df, channels, field, params):
    '''
    It returns a dataframe with the statistical operations of count, mean, std,
    min, max and sum of the values selected for each vth_t1 or vth_t2. Variable
    field can be 'vth_t1' or 'vth_t2' and params is the corresponding vth .
    '''
    df_filtered    = df[df.channel_id.isin(channels)][['channel_id', 'count']]
    operations     = ['count', 'mean', 'std', 'min', 'max', 'sum']
    df_agg         = df_filtered.groupby('channel_id').agg(operations)
    df_agg.columns = operations
    df_tmp         = df_agg.reset_index()
    df_tmp[field]  = params
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
