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

    np.savez(output_file, evt_number=np.array([i[0] for i in df0.index]),        t1=df0.t1,
                          cluster   =np.array([i[1] for i in df0.index]),        t2=df0.t2,
                          ctdaq     =df0.ctdaq,   sensor_id=df0.sensor_id,    efine=df0.efine,   efine_corrected=df0.efine_corrected,
                          ct_data   =df0.ct_data, tofpet_id=df0.tofpet_id,    tfine=df0.tfine,   tfine_corrected=df0.tfine_corrected,
                          tac_id    =df0.tac_id, channel_id=df0.channel_id,  intg_w=df0.intg_w, tcoarse_extended=df0.tcoarse_extended,
                          tcoarse   =df0.tcoarse,   ecoarse=df0.ecoarse, intg_w_ToT=df0.intg_w_ToT,       ToT_pe=df0.ToT_pe,
                          perc_cor  =df0.perc_cor)

if __name__ == "__main__":

    input_file  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    extract_charge_from_data(input_file, output_file)


## To analyze the ouput files
# df = pd.DataFrame({})
# d  = np.load(output_file, allow_pickle=True)
# for (key, vals) in d.items():
#     df[key] = vals
