import pandas as pd
import numpy  as np

from typing     import Sequence

def apply_charge_fluctuation(sns_df: pd.DataFrame, DataSiPM_idx: pd.DataFrame):
    """
    Apply a fluctuation in the total detected charge, sensor by sensor,
    according to a value read from the database.
    """
    def rand_normal(sig):
        return np.random.normal(0, sig)

    pe_resolution = DataSiPM_idx.Sigma / DataSiPM_idx.adc_to_pes
    ## The next line avoids resetting the names in the original.
    pe_resolution = pe_resolution.reset_index().rename(columns={'SensorID': 'sensor_id'})
    fluct_sns     = sns_df.join(pe_resolution.set_index('sensor_id'), on='sensor_id')
    fluct_sns.rename(columns={0:'pe_res'}, inplace=True)

    fluct_sns['charge'] += np.apply_along_axis(rand_normal, 0, fluct_sns.pe_res)

    columns    = ['event_id', 'sensor_id', 'charge']
    return fluct_sns.loc[fluct_sns.charge > 0, columns]


def apply_sipm_pde(sns_df: pd.DataFrame, pde: float) -> pd.DataFrame:
    """
    Apply a photodetection efficiency on a dataframe with sensor response.
    """
    sns_df['det_charge'] = sns_df.charge.apply(lambda x: np.count_nonzero(np.random.uniform(0, 1, x)<pde))
    sns_df = sns_df[sns_df.det_charge>0]
    sns_df = sns_df.drop(['charge'], axis=1).rename(columns={'det_charge': 'charge'})

    return sns_df
    
    



def apply_sipm_saturation(df: pd.DataFrame, rec_time: int):
    """
    This function creates a new column named 'charge' applying 
    sensor saturation taking into account their recovery time. 
    """
    def exp(x: Sequence[float], tau: int):
    	return np.exp(-x/tau)
    
    diff_time = np.diff(df.time.values)
    v_frac    = 1 - exp(diff_time, rec_time)
    charges   = np.insert(v_frac, 0, 1)
    df.insert(len(df.columns), 'charge', charges.astype(float))

    return df
