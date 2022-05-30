import sys
import pandas as pd

import antea.reco.petit_reco_functions as prf

""" To run this script:
python process_data_petit.py input_file output_file
where:
- input_file is the datafile.
- output_file is an h5 file with the filtered events.
"""

def process_data_petit(input_file, output_file):

    """
    This function selects data events in coincidences and with maximum charge
    on one of the four central SiPMs in the Hamamatsu plane. It also computes
    the charge in pe and the charge ratio of the SiPMs in the external ring over
    the total charge.
    """

    df0   = pd.DataFrame({})
    store = pd.HDFStore(input_file, 'r')
    for key in store.keys():
        df = store.get(key)
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df['intg_w_ToT'] = df['t2'] - df['t1']
        df = df[(df['intg_w_ToT']>0) & (df['intg_w_ToT']<500)]

        evt_groupby = ['evt_number', 'cluster']
        df_coinc  = prf.compute_coincidences(df, evt_groupby=evt_groupby)
        df_center = prf.select_evts_with_max_charge_at_center(df_coinc, evt_groupby=evt_groupby, tot_mode=True)

        df_center['ToT_pe'] = prf.ToT_to_pes(df_center['intg_w_ToT']*5) #### This function takes the time in ns, not in cycles!!!

        ratio_ch_corona = prf.compute_charge_ratio_in_corona(df_center, evt_groupby=evt_groupby, variable='ToT_pe')
        df_center['ratio_cor'] = ratio_ch_corona[df_center.index].values

        df0 = pd.concat([df0, df_center], ignore_index=False, sort=False)

    df    = df0.reset_index()
    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

if __name__ == "__main__":

    input_file  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    process_data_petit(input_file, output_file)
