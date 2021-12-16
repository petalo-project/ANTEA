import pandas as pd

import antea.reco.data_reco_functions as pf


def is_a_coincidence(df: pd.DataFrame) -> bool:
    """
    Returns the events in which both planes have detected charge.
    """
    sensors_d = df[df.sensor_id.unique()<100].sensor_id.nunique() # Ham
    sensors_c = df[df.sensor_id.unique()>100].sensor_id.nunique() # FBK
    return sensors_d>0 and sensors_c>0


def compute_coincidences_mc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with only coincidences.
    """
    df_filter = df.groupby('event_id').filter(is_a_coincidence)
    return df_filter


def is_max_charge_at_center(df: pd.DataFrame,
                            det_plane: bool = True,
                            variable:   str = 'charge') -> bool:
    """
    Returns True if the maximum charge of the event has been detected
    in one of the four central sensors of the desired plane.
    """
    if det_plane:
        df          = df[df.sensor_id<100]
        central_sns = pf.central_sns_det
    else:
        df          = df[df.sensor_id>100]
        central_sns = pf.central_sns_coinc

    if len(df)==0:
        return False

    argmax = df[variable].argmax()
    return df.iloc[argmax].sensor_id in central_sns


def select_evts_with_max_charge_at_center_mc(df: pd.DataFrame,
                                             det_plane: bool = True,
                                             variable:   str = 'charge') -> pd.DataFrame:
    """
    Returns a dataframe with only the events with maximum charge
    at the central sensors.
    """
    df_filter_center = df.groupby(['event_id']).filter(is_max_charge_at_center,
                                                       dropna    = True,
                                                       det_plane = det_plane,
                                                       variable  = variable)
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


def compute_charge_percentage_in_corona_mc(df: pd.DataFrame,
                                           variable: str = 'charge') -> pd.Series:
    """
    Computes the percentage of charge detected in the external corona of the detection
    plane with respect to the total charge of that plane.
    """
    tot_ch_d = df[df.sensor_id<100]            .groupby(['event_id'])[variable].sum()
    cor_ch   = df[df.sensor_id.isin(pf.corona)].groupby(['event_id'])[variable].sum()
    return (cor_ch/tot_ch_d).fillna(0)*100
