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
