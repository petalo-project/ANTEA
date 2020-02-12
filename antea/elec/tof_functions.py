import numpy  as np
import pandas as pd

from typing import Sequence, Tuple


def apply_spe_dist(time: np.array) -> Tuple[np.array, float]:
    """
    Returns a normalized array following the double exponential
    distribution of the sipm response.
    """
    spe_response = spe_dist(time)
    if np.sum(spe_response) == 0:
        return np.zeros(len(time)), 0.
    norm_dist    = np.sum(spe_response)
    spe_response = spe_response/norm_dist #Normalization
    return spe_response, norm_dist


def spe_dist(time: np.array) -> np.array:
    """
    Analitic function that calculates the double exponential decay for
    the sipm response.
    """
    alfa      = 1.0/15000
    beta      = 1.0/100
    t_p       = np.log(beta/alfa)/(beta-alfa)
    K         = (beta)*np.exp(alfa*t_p)/(beta-alfa)
    time_dist = K*(np.exp(-alfa*time)-np.exp(-beta*time))
    return time_dist


def convolve_tof(spe_response: Sequence[float],
                 signal: Sequence[float]) -> Sequence[float]:
    """
    Apply the spe_response distribution to the given signal.
    """
    if not np.count_nonzero(spe_response):
        print('spe_response values are zero')
        return np.zeros(len(spe_response)+len(signal)-1)
    conv_first = np.hstack([spe_response, np.zeros(len(signal)-1)])
    conv_res   = np.zeros(len(signal)+len(spe_response)-1)
    pe_pos     = np.argwhere(signal > 0)
    pe_recov   = signal[pe_pos]
    for i in range(len(pe_recov)): #Loop over the charges
        conv_first_ch = conv_first*pe_recov[i]
        desp          = np.roll(conv_first_ch, pe_pos[i])
        conv_res     += desp
    return conv_res


def tdc_convolution(tof_response: pd.DataFrame, spe_response: Sequence[float], time_window: float, n_sipms: int, first_sipm: int, te_tdc: float) -> Sequence[Sequence[float]]:
    """
    Apply the spe_response distribution to every sipm and returns a charge matrix of time and n_sipms dimensions.
    """
    pe_table = np.zeros((time_window, n_sipms))
    sel_tof  = tof_response[tof_response.time_bin < time_window]
    s_ids    = - sel_tof.sensor_id.values - first_sipm
    pe_table[sel_tof.time_bin.values, s_ids] = sel_tof.charge.values

    conv_table = np.zeros((len(pe_table) + len(spe_response)-1, n_sipms))
    for i in range(n_sipms):
        if np.count_nonzero(pe_table[:,i]):
            conv_table[:,i] = convolve_tof(spe_response, pe_table[0:time_window,i])
    return conv_table


def translate_charge_matrix_to_wf_df(event_id: int, conv_table: Sequence[Sequence[float]], first_sipm: int) -> pd.DataFrame:
    """
    Transform the charge matrix into a tof dataframe.
    """
    keys         = np.array(['event_id', 'sensor_id', 'time_bin', 'charge'])
    if np.all(conv_table==0):
        return pd.DataFrame({}, columns=keys)
    t_bin, s_id  = np.where(conv_table>0)
    s_id         = - s_id - first_sipm
    conv_tb_flat = conv_table.flatten()
    charge       = conv_tb_flat[conv_tb_flat>0]
    evt          = np.full(len(t_bin), event_id)
    a_wf         = np.array([evt, s_id, t_bin, charge])
    wf_df        = pd.DataFrame(a_wf.T, columns=keys)
    return wf_df
