import pandas as pd
import numpy  as np

from typing import Sequence


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


central_sns_det          = [ 44,  45,  54,  55]
central_sns_coinc_1tile  = [122, 123, 132, 133]
central_sns_coinc_4tiles = [144, 145, 154, 155]

def is_max_charge_at_center(df: pd.DataFrame,
                            det_plane:          bool = True,
                            coinc_plane_1_tile: bool = True,
                            variable:            str = 'charge',
                            tot_mode:           bool = False) -> bool:
    """
    Returns True if the maximum charge of the event has been detected
    in one of the four central sensors of the desired plane.
    """
    if det_plane:
        tofpet_id   = 0
        central_sns = central_sns_det
    else:
        tofpet_id   = 2
        if coinc_plane_1_tile:
            central_sns = central_sns_coinc_1tile
        else:
            central_sns = central_sns_coinc_4tiles

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
                                          coinc_plane_1_tile:   bool = True,
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
                                                      coinc_plane_1_tile = coinc_plane_1_tile,
                                                      variable           = variable,
                                                      tot_mode           = tot_mode)
    return df_filter_center


int_area = [22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47,
            52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77]

corona   = [11, 12, 13, 14, 15, 16, 17, 18, 21, 28, 31, 38, 41, 48,
            51, 58, 61, 68, 71, 78, 81, 82, 83, 84, 85, 86, 87, 88]

def is_event_contained_in_det_plane(df: pd.DataFrame) -> bool:
    """
    Returns True if all the sensors of the event are located within
    the internal area of the detection plane.
    """
    df          = df[df.tofpet_id == 0] ## Detection plane
    sens_unique = df.sensor_id.unique()
    if len(sens_unique):
        return set(sens_unique).issubset(set(int_area))
    else:
        return False


def select_contained_evts_in_det_plane(df: pd.DataFrame,
                                       evt_groupby: Sequence[str] = ['event_id']) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with touched sensors
    located within the internal area of the detection plane.
    """
    df_cov_evts = df.groupby(evt_groupby).filter(is_event_contained_in_det_plane)
    return df_cov_evts


def compute_charge_ratio_in_corona(df: pd.DataFrame,
                                   evt_groupby: Sequence[str] = ['event_id'],
                                   variable: str = 'charge') -> pd.Series:
    """
    Computes the ratio of charge detected in the external corona of the detection
    plane with respect to the total charge of that plane.
    """
    tot_ch_d = df[df.tofpet_id==0]          .groupby(evt_groupby)[variable].sum()
    cor_ch   = df[df.sensor_id.isin(corona)].groupby(evt_groupby)[variable].sum()
    return (cor_ch/tot_ch_d).fillna(0)
