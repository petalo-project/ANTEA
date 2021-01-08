import numpy  as np
import pandas as pd


def apply_sipm_pde(sns_df: pd.DataFrame, pde: float) -> pd.DataFrame:
    """
    Apply a photodetection efficiency on a dataframe with sensor response.
    """
    sns_df['det_charge'] = sns_df.charge.apply(lambda x: np.count_nonzero(np.random.uniform(0, 1, x)<pde))
    sns_df = sns_df[sns_df.det_charge>0]
    sns_df = sns_df.drop(['charge'], axis=1).rename(columns={'det_charge': 'charge'})

    return sns_df
