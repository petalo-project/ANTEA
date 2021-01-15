import pandas as pd
import numpy  as np

from invisible_cities.reco.sensor_functions import charge_fluctuation

def apply_charge_fluctuation(sns_df: pd.DataFrame, DataSiPM_idx: pd.DataFrame):
    """
    Apply a fluctuation in the total detected charge, sensor by sensor,
    according to a value read from the database.
    """

    sum_sns = sns_df.groupby(['event_id','sensor_id'])[['charge']].sum()
    sum_sns = sum_sns.reset_index()
    charges = sum_sns.charge.values
    sipmrd  = charges[:, np.newaxis] # IC function expects a waveform

    pe_resolution = DataSiPM_idx.Sigma / DataSiPM_idx.adc_to_pes
    touched_sipms = sum_sns.sensor_id.values
    pe_res        = pe_resolution[touched_sipms]

    sipm_fluct = np.array(tuple(map(charge_fluctuation, sipmrd, pe_res)))

    events      = sns_df.event_id.unique()
    sns_per_evt = sum_sns.event_id.value_counts()
    instances   = sns_per_evt[events]

    evts         = np.repeat(events, instances)
    t_bins       = np.repeat(0, len(sipmrd))
    fluct_charge = sipm_fluct.flatten()
    fluct_sns    = pd.DataFrame({'event_id': evts, 'sensor_id': touched_sipms,
                                 'time_bin': t_bins, 'charge': fluct_charge})

    return fluct_sns


def apply_sipm_pde(sns_df: pd.DataFrame, pde: float) -> pd.DataFrame:
    """
    Apply a photodetection efficiency on a dataframe with sensor response.
    """
    sns_df['det_charge'] = sns_df.charge.apply(lambda x: np.count_nonzero(np.random.uniform(0, 1, x)<pde))
    sns_df = sns_df[sns_df.det_charge>0]
    sns_df = sns_df.drop(['charge'], axis=1).rename(columns={'det_charge': 'charge'})

    return sns_df
