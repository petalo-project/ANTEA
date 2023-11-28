import pandas as pd
import numpy  as np

from typing import Sequence, Tuple


def tofpetid(sid: int) -> int:
    """
    Returns 0 if sensor_id is below 100 (detector plane)
    and 2 if not (coinc plane).
    """
    if sid < 100: return 0
    else: return 2


def ToT_to_pes(x: float) -> float:
    """
    Function that translate ToT from PETSYS to pes according
    to a exponential relationship. x units are ns.
    """
    return 9.98597793 * np.exp(x/252.69045094)


def compute_coincidences(df: pd.DataFrame,
                         evt_groupby: Sequence[str] = ['event_id']) -> pd.DataFrame:
    """
    Returns the events in which both planes have detected charge.
    """
    nplanes  = df.groupby(evt_groupby)['tofpet_id'].nunique()
    df_idx   = df.set_index(evt_groupby)
    df_coinc = df_idx.loc[nplanes[nplanes == 2].index]

    return df_coinc


def sensor_params(det_plane:          bool = True,
                  coinc_plane_4tiles: bool = False) -> Tuple[int, Sequence[int], Sequence[int], Sequence[int]]:
    """
    Returns the corresponding ids of tofpet_id, central_sns, int_area and corona
    of the desired plane.
    """
    tofpet_id   = 0
    central_sns = np.array([ 44,  45,  54,  55])
    int_area    = np.array([22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47,
                            52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77])
    corona      = np.array([11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
                            51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88])
    if not det_plane:
        tofpet_id = 2
        if coinc_plane_4tiles:
            central_sns = central_sns + 100
            int_area    = int_area    + 100
            corona      = corona      + 100
        else:
            central_sns = np.array([122, 123, 132, 133])
            int_area    = central_sns
            corona      = np.array([111, 112, 113, 114, 121, 124, 131, 134, 141, 142, 143, 144])

    return tofpet_id, central_sns, int_area, corona


def is_max_charge_at_center(df: pd.DataFrame,
                            det_plane:          bool = True,
                            coinc_plane_4tiles: bool = False,
                            variable:            str = 'charge',
                            tot_mode:           bool = False) -> bool:
    """
    Returns True if the maximum charge of the event has been detected
    in one of the four central sensors of the desired plane.
    """
    tofpet_id, central_sns, _, _ = sensor_params(det_plane, coinc_plane_4tiles)

    df = df[df.tofpet_id == tofpet_id]
    if len(df)==0:
        return False

    if tot_mode: # t2 - t1 instead of intg_w or efine_corrected
        argmax = (df.t2 - df.t1).argmax()
    else:
        argmax = df[variable].argmax()

    return df.iloc[argmax].sensor_id in central_sns


def select_evts_with_max_charge_at_center(df: pd.DataFrame,
                                          evt_groupby: Sequence[str] = ['event_id'],
                                          det_plane:            bool = True,
                                          coinc_plane_4tiles:   bool = False,
                                          variable:              str = 'charge',
                                          tot_mode:             bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with maximum charge
    at the central sensors. If MC is being analyzed, `variable`
    should be `charge` and `tot_mode` False.
    """
    df_filter_center = df.groupby(evt_groupby).filter(is_max_charge_at_center,
                                                      dropna             = True,
                                                      det_plane          = det_plane,
                                                      coinc_plane_4tiles = coinc_plane_4tiles,
                                                      variable           = variable,
                                                      tot_mode           = tot_mode)
    return df_filter_center


def is_event_contained(df: pd.DataFrame,
                       det_plane:          bool = True,
                       coinc_plane_4tiles: bool = False) -> bool:
    """
    Returns True if all the sensors of the event are located within
    the internal area of the desired plane.
    """
    tofpet_id, _, int_area, _ = sensor_params(det_plane, coinc_plane_4tiles)

    df          = df[df.tofpet_id == tofpet_id]
    sens_unique = df.sensor_id.unique()
    if len(sens_unique):
        return set(sens_unique).issubset(set(int_area))
    else:
        return False


def select_contained_evts(df: pd.DataFrame,
                          evt_groupby: Sequence[str] = ['event_id'],
                          det_plane:            bool = True,
                          coinc_plane_4tiles:   bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with touched sensors
    located within the internal area of the desired plane.
    """
    df_cov_evts = df.groupby(evt_groupby).filter(is_event_contained,
                                                 dropna             = True,
                                                 det_plane          = det_plane,
                                                 coinc_plane_4tiles = coinc_plane_4tiles)
    return df_cov_evts


def compute_charge_ratio_in_corona(df: pd.DataFrame,
                                   evt_groupby: Sequence[str] = ['event_id'],
                                   variable:              str = 'charge',
                                   det_plane:            bool = True,
                                   coinc_plane_4tiles:   bool = False) -> pd.Series:
    """
    Computes the ratio of charge detected in the external corona of the desired
    plane with respect to the total charge of that plane.
    """
    tofpet_id, _, _, corona =  sensor_params(det_plane, coinc_plane_4tiles)

    tot_ch_d = df[df.tofpet_id==tofpet_id]  .groupby(evt_groupby)[variable].sum()
    cor_ch   = df[df.sensor_id.isin(corona)].groupby(evt_groupby)[variable].sum()
    return (cor_ch/tot_ch_d).fillna(0)
    
    
def compute_charge_in_groups_of_4sensors(df): 
    
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

            df = df.replace({'sensor_id':sns_comp}, new_sns[i])

            if (i == 3) | (i == 7) | (i == 11): # To change the row in a tile
                j += 10
            else:
                j += 2
        j = 0

    return df
    
