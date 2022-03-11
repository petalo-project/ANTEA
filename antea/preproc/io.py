import pandas as pd
import tables as tb
import numpy  as np
import os

from glob import glob


def compute_file_chunks_indices(filein, chunk_size = 500000):
    '''
    It returns an array with indices separating data in chunks without
    splitting the data with same event number.
    '''
    with tb.open_file(filein) as h5in:
        evt_numbers = h5in.root.data.cols.evt_number
        evt_diffs   = np.diff(evt_numbers)
        evt_limits  = np.concatenate([np.where(evt_diffs)[0],
                                      np.array(evt_numbers.shape)])

        # Find borders that keep ~chunk_size rows per chunk
        chunk_diffs  = np.diff(evt_limits // chunk_size)
        chunk_limits = np.where(chunk_diffs)[0]

        chunks = np.concatenate([np.array([0]), evt_limits[chunk_limits],
                                 np.array(evt_numbers.shape)])
        return chunks


def write_corrected_df_daq(fileout, df, iteration, append=False):
    '''
    It writes the data in an output file. If it does not exist, it creates it and
    if it exists, it appends new data.
    '''
    table_name = 'data_{}'.format(iteration)
    mode = 'a' if append else 'w'
    store = pd.HDFStore(fileout, mode, complib=str("zlib"), complevel=4)
    store.put(table_name, df, index=False, format='table', data_columns=None)
    store.close()


def get_files(run, folder = '/analysis/{run}/hdf5/data/'):
    '''
    It returns the name of all files for a given run number
    '''
    pattern = os.path.join(folder.format(run = run), '*h5')
    files = glob(pattern)
    files.sort()
    return files


def read_run_data(files, verbose=False):
    '''
    It returns a dataframe with data from all given files
    '''
    dfs = []
    for i, fname in enumerate(files):
        if verbose:
            print(i, fname)
        try:
            df = pd.read_hdf(fname, 'data')
            df['fileno'] = i
            dfs.append(df)
        except:
            print("Error in file ", fname)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def get_evt_times(files, verbose=False):
    '''
    It returns a dataframe with a column with time calculation in date format
    and a column with time differences between consecutive events.
    '''
    time_dfs = []

    for i, fname in enumerate(files):
        if verbose:
            print(i, fname)
        df_time = pd.read_hdf(fname, 'dateEvents')
        df_time['fileno'] = i
        time_dfs.append(df_time)
    df_times = pd.concat(time_dfs)
    df_times['date'] = pd.to_datetime(df_times.timestamp, unit='us')

    # Compute time difference between one event and the next
    df_times['time_diff'] = np.abs((df_times.timestamp/1e6).diff(periods=-1))
    df_times = df_times.fillna(0)
    return df_times
