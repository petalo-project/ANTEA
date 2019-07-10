import numpy  as np
import pandas as pd

from typing import Sequence, Tuple


def lower_or_equal(f1: float, f2: float, allowed_error: float = 1.e-6) -> bool:
    return f1 <= f2 + allowed_error


def greater_or_equal(f1: float, f2: float, allowed_error: float = 1.e-6) -> bool:
    return f1 >= f2 - allowed_error


def from_cartesian_to_cyl(pos: Sequence[np.array]) -> Sequence[np.array]:
    cyl_pos = np.array([np.sqrt(pos[:,0]**2+pos[:,1]**2), np.arctan2(pos[:,1], pos[:,0]), pos[:,2]]).transpose()
    return cyl_pos


def phi_mean_var(pos_phi: Sequence[float], q: Sequence[float]) -> float:
    diff_sign = min(pos_phi ) < 0 < max(pos_phi)
    if diff_sign & (np.abs(np.min(pos_phi))>np.pi/2):
        pos_phi[pos_phi<0] = np.pi + np.pi + pos_phi[pos_phi<0]
    mean_phi = np.average(pos_phi, weights=q)
    var_phi  = np.average((pos_phi-mean_phi)**2, weights=q)

    return mean_phi, var_phi


def find_SiPMs_over_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Integrate the charge in time of each SiPM and select only those with total
    charge larger than threshold.
    """
    tot_charges_df = df.groupby(['event_id','sensor_id'])[['charge']].sum()
    return tot_charges_df[tot_charges_df.charge > threshold].reset_index()


def find_closest_sipm(point: Tuple[float, float, float], sipms: pd.DataFrame) -> pd.DataFrame:
   """
   Find the closest SiPM to a point, given a df of SiPMs.
   """
   sns_positions = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()

   subtr        = np.subtract(point, sns_positions)
   distances    = [np.linalg.norm(d) for d in subtr]
   min_dist     = np.min(distances)
   min_sipm     = np.isclose(distances, min_dist)
   closest_sipm = sipms[min_sipm]

   return closest_sipm


def assign_sipms_to_gammas(waveforms: pd.DataFrame, true_pos: Sequence[Tuple[float, float, float]], DataSiPM_idx: pd.DataFrame) -> Tuple[Sequence[float], Sequence[float], Sequence[Tuple[float, float, float]], Sequence[Tuple[float, float, float]]]:
    """
    Divide the SiPMs with charge between the two back-to-back gammas,
    or to one of the two if the other one hasn't interacted.
    Return the lists of the charges and the positions of the SiPMs of the two groups.
    """
    sipms           = DataSiPM_idx.loc[waveforms.sensor_id]
    sns_closest_pos = [np.array([find_closest_sipm(pos, sipms).X.values, find_closest_sipm(pos, sipms).Y.values, find_closest_sipm(pos, sipms).Z.values]).transpose()[0] for pos in true_pos]

    q1, q2     = [], []
    pos1, pos2 = [], []

    sns_charges = waveforms.charge
    closest_pos = sns_closest_pos[0] ## Look at the first one, which always exists.
    ### The sensors on the same semisphere are grouped together,
    ### and those on the opposite side, too, only
    ### if two interactions have been detected.
    for sns_pos, charge in zip(sns_positions, sns_charges):
        scalar_prod = sum(a*b for a, b in zip(sns_pos, closest_pos))
        if scalar_prod > 0.:
            q1.append(charge)
            pos1.append(sns_pos)
        elif len(sns_closest_pos) == 2:
            q2.append(charge)
            pos2.append(sns_pos)

    return q1, q2, pos1, pos2
