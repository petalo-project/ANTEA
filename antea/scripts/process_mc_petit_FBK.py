import os, sys
import numpy  as np
import pandas as pd


from invisible_cities.core import system_of_units as units

import antea.database.load_db          as db
import antea.mcsim.sensor_functions    as snsf
import antea.elec.shaping_functions    as shf
import antea.reco.petit_reco_functions as prf
import antea.reco.reco_functions       as rf

from antea.core.exceptions import WaveformEmptyTable
from antea.io.mc_io        import load_mcTOFsns_response

from typing     import Sequence


""" To run this script:
python process_data_petit_FBK_MC.py input_file output_file recovery_time
where:
- input_file is the datafile
- output_file if an h5 file with the filtered events with and
  without saturation applied for the FBK SiPMs.
- recovery_time is the time that FBK SiPMs take to recover. 
"""

def num_sensors_4in4(df): 
    
    """
    This function replaces the sensor_id in the data with 
    the ordering usually used: TOFPET 0 from 11 to 88 and TOFPET 2
    from 111 to 188; tacking into account the combination of 4 SiPMs, 
    as they are read 4 by 4 in the real set-up.
    """

    j = 0 # To go through the sensors of a tile.
    n = 0 # To change the tile
    
    num_tiles   = 8
    num_sensors = 16

    new_sns_1 = np.array([11, 12, 13, 14, 
                          21, 22, 23, 24, 
                          31, 32, 33, 34, 
                          41, 42, 43, 44])

    for k in np.arange(num_tiles):

        if   k == 0:
            new_sns = new_sns_1       # tile 1
        elif k == 1:
            new_sns = new_sns_1 + 4   # tile 2
        elif k == 2:
            new_sns = new_sns_1 + 40  # tile 3
        elif k == 3:
            new_sns = new_sns_1 + 44  # tile 4
        elif k == 4:
            new_sns = new_sns_1 + 100 # tile 5
        elif k == 5:
            new_sns = new_sns_1 + 104 # tile 6
        elif k == 6:
            new_sns = new_sns_1 + 140 # tile 7
        elif k == 7:
            new_sns = new_sns_1 + 144 # tile 8       

        n = k*100

        for i in np.arange(num_sensors):

            sns_comp = (np.array([100, 101, 108, 109]) + j + n).tolist()

            if n < 390: # To change the TOFPET plane
                df = df.replace({'sensor_id':sns_comp}, new_sns[i])

            else:
                df = df.replace({'sensor_id':sns_comp}, new_sns[i])

            if (i == 3) | (i == 7) | (i == 11): # To change the row in a tile
                j += 10
            else:
                j += 2
        j = 0

    return df


def exp(x: Sequence[float], tau: int):
    
    """ 
    This function returns the exponential for the given data.
    """
    return np.exp(-x/tau)


def sim_saturation_vicente(df: pd.DataFrame, rec_time: int):
    """
    This function creates a new column named 'charge' applying 
    sensor saturation taking into account their recovery time. 
    """
    
    diff_time = np.diff(df.time.values)
    v_frac    = 1 - exp(diff_time, rec_time)
    charges   = np.insert(v_frac, 0, 1)
    df.insert(len(df.columns), 'charge', charges.astype(float))

    return df
    

def process_mc_petit_FBK(input_file: str, output_file: str, recovery_time: int)

    """
    This function selects data events in coincidence and it saves the maximum
    chargeof the event in each the plane and the minimum time. It also computes
    the charge obtained in a SiPM with and without saturation. Finally, it sums 
    the charge of 4 in 4 sensors to compare with real data.
    """
    
    DataSiPM_pb     = db.DataSiPM('petalo', 12334, 'PB')
    DataSiPM_pb_idx = DataSiPM_pb.set_index('SensorID')

    ### parameters for single photoelectron convolution in SiPM response
    tau_sipm       = [100, 15000]
    time_window    = 5000
    time           = np.arange(0, time_window)
    spe_resp, norm = shf.normalize_sipm_shaping(time, tau_sipm)

    n_pe          = 1  # number of first photoelectron to be considered for times
    sigma_sipm    = 40 #ps SiPM jitter
    sigma_elec    = 30 #ps electronic jitter
    timestamp_thr = 0.25 # PETsys threshold to extract the timestamp

    start = int(sys.argv[1])
    numb  = int(sys.argv[2])
    rec_time = float(sys.argv[3])

    
    ### Data to save
    sensor_id0, sensor_id2   = [], []
    max_charge0, max_charge2 = [], []
    max_or0, max_or2         = [], []
    tmin0, tmin2             = [], []
    tmin_ch0, tmin_ch2       = [], []
    tmin_sid0, tmin_sid2     = [], []
    event_ids                = []

    
    try:
        tof_response = load_mcTOFsns_response(input_file)
    except ValueError:
        print(f'File {input_file} not found')
        exit()
    except OSError:
        print(f'File {input_file} not found')
        exit()
    except KeyError:
        print(f'No object named MC/tof_sns_response in file {input_file}'')
        exit()
    except:
        print(f'Unknown error in {input_file}')
        exit()
    print(f'Analyzing file {input_file}')


    ### Add column for real sensor_id instead of cells_id          
    tof_response = tof_response.rename(columns={'sensor_id':'sns_cells'})
    tof_sensors  = tof_response.sns_cells.values // 10000
    tof_response['sensor_id'] = tof_sensors
    tof_response = num_sensors_4in4(tof_response)

    ### Apply saturation:
    events = tof_response.event_id.unique()
    n_evts           = len(events)
    n_evts_per_bunch = int(100)
    n_bunches        = int(np.ceil(n_evts / n_evts_per_bunch))
    print(n_evts, n_bunches)
    for n in range(n_bunches):
        print('bunch', n)
        evt_range = events[n*n_evts_per_bunch:(n+1)*n_evts_per_bunch]
        b_sns   = tof_response[tof_response.event_id.isin(evt_range)]
        charge_df = b_sns.groupby(['event_id','sns_cells'], as_index=False).apply(sim_saturation_vicente, rec_time=rec_time)

        ### Create DataFrame for charge data, with and without saturation:
        print('creating charge dataFrames')
        charge_df.insert(len(charge_df.columns), 'original_pe', np.ones(len(charge_df.sensor_id.values)).astype(int))

        # With saturation
        tot_sns    = charge_df.groupby(['event_id', 'sensor_id'])['charge'].sum()
        tot_sns_df = pd.DataFrame(tot_sns)
        tot_a      = tot_sns_df.reset_index()
        
        # Without saturation
        orig_sns    = charge_df.groupby(['event_id', 'sensor_id'])['original_pe'].sum()
        orig_sns_df = pd.DataFrame(orig_sns)
        orig_a      = orig_sns_df.reset_index()

        #Sum charge 4 in 4 SiPMs:
        tot  = tot_a.groupby(['event_id', 'sensor_id'])['charge'].sum().reset_index()
        orig = orig_a.groupby(['event_id', 'sensor_id'])['original_pe'].sum().reset_index()

        # Adding charge fluctuation and charge threshold
        tot = snsf.apply_charge_fluctuation(tot, DataSiPM_pb_idx)
        tot = rf.find_SiPMs_over_threshold(tot, threshold=2)
        tot.insert(len(tot.columns), 'original_pe', orig.original_pe.astype(float))

        tot['tofpet_id'] = tot['sensor_id'].apply(prf.tofpetid)

        sns_coinc = prf.compute_coincidences(tot, evt_groupby=['event_id'])
        sns_coinc = sns_coinc.reset_index()

        c_events = sns_coinc.event_id.unique()
        print(f'Number of events = {len(events)}')

        for evt in c_events:
            evt_sns = sns_coinc[sns_coinc.event_id == evt]
            evt_tof = tof_response[tof_response.event_id == evt]

            if (len(evt_sns) == 0) | (len(evt_tof) == 0):
                continue

            evt_tof = evt_tof[evt_tof.sensor_id.isin(evt_sns.sensor_id)]

            if len(evt_tof) == 0:
                continue

            ### Divide sensors
            df_sns0 = evt_sns[evt_sns.sensor_id < 100]
            df_sns2 = evt_sns[evt_sns.sensor_id > 100]

            sns0 = df_sns0.sensor_id.values
            sns2 = df_sns2.sensor_id.values

            ### Use absolute times in units of ps
            times = evt_tof.time / units.ps
              
            ### Add SiPM jitter, if different from zero
            if sigma_sipm > 0:
                times = np.round(np.random.normal(times, sigma_sipm))
            evt_tof = evt_tof.drop('time', axis=1)
            evt_tof.insert(len(evt_tof.columns), 'time', np.round(times).astype(int)) # here we have bins of 1 ps
            evt_tof.insert(len(evt_tof.columns), 'charge', np.ones(len(times)).astype(int))

            ### Produce a TOF dataframe with convolved time response
            tof_sns = evt_tof.sensor_id.unique()

            evt_tof_exp_dist = []
            for s_id in tof_sns:
                tdc_conv    = shf.sipm_shaping_convolution(evt_tof, spe_resp, s_id, time_window)
                tdc_conv_df = shf.build_convoluted_df(evt, s_id, tdc_conv)

                if sigma_elec > 0:
                    tdc_conv_df = tdc_conv_df.assign(time=np.random.normal(tdc_conv_df.time.values, sigma_elec))

                tdc_conv_df = tdc_conv_df[tdc_conv_df.charge > timestamp_thr/norm]
                tdc_conv_df = tdc_conv_df[tdc_conv_df.time == tdc_conv_df.time.min()]
                evt_tof_exp_dist.append(tdc_conv_df)
            evt_tof_exp_dist = pd.concat(evt_tof_exp_dist)


            try:
                min_id0, min_t0 = rf.find_first_time_of_sensors(evt_tof_exp_dist, sns0, n_pe)
            except WaveformEmptyTable:
                print(f'TOF dataframe has no minimum time in plane 0 for event {evt}')
                min_id0, min_t0 = [-1], -1
            try:
                min_id2, min_t2 = rf.find_first_time_of_sensors(evt_tof_exp_dist, sns2, n_pe)
            except:
                print(f'TOF dataframe has no minimum time in plane 2 for event {evt}')
                min_id2, min_t2 = [-1], -1

            ## select SiPM with max charge
            max_ch0 = df_sns0.loc[df_sns0.charge.idxmax()]
            max_ch2 = df_sns2.loc[df_sns2.charge.idxmax()]
            sid_0   = max_ch0.sensor_id
            sid_2   = max_ch2.sensor_id
            ch_0    = max_ch0.charge
            ch_2    = max_ch2.charge
            or_0    = max_ch0.original_pe
            or_2    = max_ch2.original_pe

            t_ch0 = evt_tof_exp_dist[evt_tof_exp_dist.sensor_id == sid_0].time.min()
            t_ch2 = evt_tof_exp_dist[evt_tof_exp_dist.sensor_id == sid_2].time.min()
            tmin_ch0.append(t_ch0)
            tmin_ch2.append(t_ch2)
            max_charge0.append(ch_0)
            max_charge2.append(ch_2)
            max_or0.append(or_0)
            max_or2.append(or_2)
            sensor_id0.append(sid_0)
            sensor_id2.append(sid_2)
            tmin0.append(min_t0)
            tmin2.append(min_t2)
            tmin_sid0.append(min_id0[0])
            tmin_sid2.append(min_id2[0])
            event_ids.append(evt)


    df = pd.DataFrame({'event_id': event_ids, 'max_charge0': max_charge0, 'max_charge2': max_charge2,
                       'max_or0' : max_or0, 'max_or2' : max_or2,
                       'sns_id0': sensor_id0, 'sns_id2': sensor_id2,
                       'tmin0': tmin0, 'tmin2': tmin2,
                       'tmin_ch0': tmin_ch0, 'tmin_ch2': tmin_ch2,
                       'tmin_sns_id0': tmin_sid0, 'tmin_sns_id2': tmin_sid2})

    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('analysis', df, format='table', data_columns=True)
    store.close()
 
if __name__ == "__main__":
              
    input_file    = str(sys.argv[1])
    output_file   = str(sys.argv[2])
    recovery_time = int(sys.argv[3])
    process_data_petit_FBK_MC(input_file, output_file, recovery_time)
    
