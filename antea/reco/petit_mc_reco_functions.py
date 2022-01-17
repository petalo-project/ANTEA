import pandas as pd


def tofpetid(sid: int) -> int:
    """
    Returns 0 if sensor_id is below 100 (detector plane)
    and 2 if not (coinc plane).
    """
    if sid < 100: return 0
    else: return 2


def params(df: pd.DataFrame, data_or_mc: str = 'mc'):

    if data_or_mc == 'mc':
        df['tofpet_id'] = df['sensor_id'].apply(tofpetid)
        evt_groupby     = ['event_id']
    elif data_or_mc == 'data':
        evt_groupby     = ['evt_number', 'cluster']
    else:
        raise ValueError(f"Unrecognized data_or_mc type: {data_or_mc}.")

    return df, evt_groupby


def compute_coincidences(df: pd.DataFrame, data_or_mc: str = 'mc') -> pd.DataFrame:
    """
    Returns the events in which both planes have detected charge.
    """
    df, evt_groupby = params(df, data_or_mc)
    nplanes  = df.groupby(evt_groupby)['tofpet_id'].nunique()
    df_idx   = df.set_index(evt_groupby)
    df_coinc = df_idx.loc[nplanes[nplanes == 2].index]

    return df_coinc


central_sns_det   = [ 44,  45,  54,  55]
central_sns_coinc = [122, 123, 132, 133]

def is_max_charge_at_center(df: pd.DataFrame,
                            det_plane: bool = True,
                            variable:   str = 'charge',
                            tot_mode:  bool = False) -> bool:
    """
    Returns True if the maximum charge of the event has been detected
    in one of the four central sensors of the desired plane.
    """
    if det_plane:
        tofpet_id   = 0
        central_sns = central_sns_det
    else:
        tofpet_id   = 2
        central_sns = central_sns_coinc

    df = df[df.tofpet_id == tofpet_id]
    if len(df)==0:
        return False

    if tot_mode: # t2 - t1 instead of intg_w or efine_corrected
        argmax = (df.t2 - df.t1).argmax()
    else:
        argmax = df[variable].argmax()

    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id in central_sns


def select_evts_with_max_charge_at_center(df: pd.DataFrame,
                                          data_or_mc: str = 'mc',
                                          det_plane: bool = True,
                                          variable:   str = 'charge',
                                          tot_mode:  bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with maximum charge
    at the central sensors. If MC is being analyzed, `variable`
    should be `charge` and `tot_mode` False.
    """
    df, evt_groupby  = params(df, data_or_mc)
    df_filter_center = df.groupby(evt_groupby).filter(is_max_charge_at_center,
                                                      dropna    = True,
                                                      det_plane = det_plane,
                                                      variable  = variable,
                                                      tot_mode  = tot_mode)
    return df_filter_center


def is_event_contained_in_det_plane(df: pd.DataFrame) -> bool:
    """
    Returns True if all the sensors of the event are located within
    the internal area of the detection plane.
    """
    df = df[df.sensor_id<100] ## Detection plane
    sens_unique = df.sensor_id.unique()
    if len(sens_unique):
        return set(sens_unique).issubset(set(pf.int_area))
    else:
        return False


def select_contained_evts_in_det_plane_mc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with only the events with touched sensors
    located within the internal area of the detection plane.
    """
    df_cov_evts = df.groupby(['event_id']).filter(is_event_contained_in_det_plane)
    return df_cov_evts


def compute_charge_ratio_in_corona_mc(df: pd.DataFrame,
                                           variable: str = 'charge') -> pd.Series:
    """
    Computes the ratio of charge detected in the external corona of the detection
    plane with respect to the total charge of that plane.
    """
    tot_ch_d = df[df.sensor_id<100]            .groupby(['event_id'])[variable].sum()
    cor_ch   = df[df.sensor_id.isin(pf.corona)].groupby(['event_id'])[variable].sum()
    return (cor_ch/tot_ch_d).fillna(0)*100
