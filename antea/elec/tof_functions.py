import numpy  as np
import pandas as pd

from typing import Sequence, Tuple


def spe_dist(time: Sequence[float]) -> Sequence[float]:
    """
    Double exponential decay for the sipm response. Returns a normalized array.
    """
    alfa         = 1.0/15000
    beta         = 1.0/100
    t_p          = np.log(beta/alfa)/(beta-alfa)
    K            = (beta)*np.exp(alfa*t_p)/(beta-alfa)
    spe_response = K*(np.exp(-alfa*time)-np.exp(-beta*time))
    if np.sum(spe_response) == 0:
        return np.zeros(len(time))
    spe_response = spe_response/np.sum(spe_response) #Normalization
    return spe_response


def convolve_tof(spe_response: Sequence[float], signal: Sequence[float]) -> Sequence[float]:
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
    pe_table  = np.zeros((time_window, n_sipms))
    for i, wf in tof_response.iterrows():
        if wf.time_bin < time_window:
            s_id = - wf.sensor_id - first_sipm
            pe_table[wf.time_bin, s_id] = wf.charge

    conv_table = np.zeros((len(pe_table) + len(spe_response)-1, n_sipms))
    for i in range(n_sipms):
        if np.count_nonzero(pe_table[:,i]):
            conv_table[:,i] = convolve_tof(spe_response, pe_table[0:time_window,i])
    return conv_table


def translate_charge_matrix_to_wf_df(event_id: int, conv_table: Sequence[Sequence[float]], first_sipm: int) -> pd.DataFrame:
    """
    Transform the charge matrix into a tof dataframe.
    """
    list_wf = []
    for t in range(len(conv_table)):
        for s_id in range(conv_table.shape[1]):
            charge = conv_table[t,s_id]
            s_id = - s_id - first_sipm
            if charge > 0.:
                list_wf.append(np.array([event_id, s_id, t, charge]))
    a_wf  = np.array(list_wf)
    keys  = np.array(['event_id', 'sensor_id', 'time_bin', 'charge'])
    wf_df = pd.DataFrame(a_wf, columns = keys)
    return wf_df
