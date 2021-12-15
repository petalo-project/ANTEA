import sys
import numpy  as np
import pandas as pd

import antea.reco.data_reco_functions as drf


def from_ToT_to_pes(x):
    return 9.98597793 * np.exp(x/252.69045094)

def extract_charge_from_data(input_file, output_file):

    df0   = pd.DataFrame({})
    store = pd.HDFStore(input_file, 'r')
    for key in store.keys():
        print(key)
        df = store.get(key)
        df = df[df.cluster != -1] ## Filtering events with only one sensor

        df['intg_w_ToT'] = df['t2'] - df['t1']
        df = df[(df['intg_w_ToT']>0) & (df['intg_w_ToT']<500)]

        df_coinc  = drf.compute_coincidences(df)
        df_center = drf.select_evts_with_max_charge_at_center(df_coinc, tot_mode=True)
        df_center['ToT_pe'] = from_ToT_to_pes(df_center['intg_w_ToT']*5) #### This function takes the time in ns, not in cycles!!!

        perc_ch_corona = drf.compute_charge_percentage_in_corona(df_center)
        df_center['perc_cor'] = perc_ch_corona[df_center.index].values

        df0 = pd.concat([df0, df_center], ignore_index=False, sort=False)

    df    = df0.reset_index()
    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('data', df, format='table', data_columns=True)
    store.close()

if __name__ == "__main__":

    input_file  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    extract_charge_from_data(input_file, output_file)
