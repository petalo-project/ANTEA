import pandas as pd


def compute_coincidences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the events in which both planes have detected charge.
    """
    nplanes = df.groupby(['evt_number', 'cluster'])['tofpet_id'].nunique()
    df_idx  = df.set_index(['evt_number', 'cluster'])
    df_coincidences = df_idx.loc[nplanes[nplanes == 2].index]

    return df_coincidences


def filter_evt_with_max_charge_at_center(df: pd.DataFrame,
                                         det_plane: bool = True,
                                         variable: str = 'efine_corrected',
                                         tot_mode: bool = False) -> bool:
    """
    Returns True if the maximum charge of the event has been detected in one of the four central sensors of the desired plane.
    """
    if det_plane:
        tofpet_id   = 0
        central_sns = [44, 45, 54, 55]
    else:
        tofpet_id   = 2
        central_sns = [122, 123, 132, 133]

    df = df[df.tofpet_id == tofpet_id]
    if len(df)==0:
        return False

    if tot_mode: # t2 - t1 instead of intg_w or efine_corrected
        argmax = (df.t2 - df.t1).argmax()
    else:
        argmax = df[variable].argmax()

    return df.iloc[argmax].sensor_id in central_sns


def select_evts_with_max_charge_at_center(df: pd.DataFrame,
                                         det_plane: bool = True,
                                         variable: str = 'efine_corrected',
                                         tot_mode: bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with maximum charge at the central sensors.
    """
    df_filter_center = df.groupby(['evt_number', 'cluster']).filter(filter_evt_with_max_charge_at_center, dropna=True, det_plane=det_plane, variable=variable, tot_mode=tot_mode)
    
    return df_filter_center


int_area = [22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77]

def filter_covered_evt(df: pd.DataFrame, min_sns: int = 2) -> bool:
    """
    Returns True if all the sensors of the event are located within
    the internal area of the detection plane. The minimun number of
    touched sensors is min_sns, 2 by default.
    """
    df = df[df.tofpet_id == 0] ## Detection plane
    sens_unique = df.sensor_id.unique()
    if len(sens_unique) >= min_sns:
        return set(sens_unique).issubset(set(int_area))
    else:
        return False


def select_covered_evts(df: pd.DataFrame, min_sns: int = 2) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with touched sensors
    located within the internal area of the detection plane.
    """
    df_cov_evts = df.groupby(['evt_number', 'cluster']).filter(filter_covered_evt,
                                                               dropna = True,
                                                               min_sns = min_sns)
    return df_cov_evts


