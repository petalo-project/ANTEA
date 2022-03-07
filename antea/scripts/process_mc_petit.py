import sys
import numpy  as np
import pandas as pd

import antea.reco.petit_reco_functions as prf

import antea.io.mc_io as mcio

""" To run this script:
python process_mc_petit.py input_file output_file
where:
- input_file is the datafile.
- output_file is an h5 file with the filtered events.
"""


def process_mc_petit(input_file, output_file):

    """
    This function selects MC events in coincidences and with maximum charge
    on one of the four central SiPMs in the Hamamatsu plane. It also computes
    the charge in pe and the charge ratio of the SiPMs in the external ring over
    the total charge.
    """

    df = mcio.load_mcsns_response(input_file)

    evt_groupby     = ['event_id']
    df['tofpet_id'] = df['sensor_id'].apply(prf.tofpetid)

    df_coinc  = prf.compute_coincidences(df, evt_groupby=evt_groupby)
    df_center = prf.select_evts_with_max_charge_at_center(df_coinc, evt_groupby=evt_groupby, variable='charge')

    ratio_ch_corona = prf.compute_charge_ratio_in_corona(df_center, evt_groupby=evt_groupby, variable='charge')
    df_center['ratio_cor'] = ratio_ch_corona[df_center.index].values

    df    = df_center.reset_index()
    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('mc', df, format='table', data_columns=True)
    store.close()

if __name__ == "__main__":

    input_file  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    process_data_petit(input_file, output_file)
