import numpy  as np
import pandas as pd

from antea.core .exceptions    import WaveformEmptyTable
from antea.utils.map_functions import Map

from typing import Sequence, Tuple


def lower_or_equal(f1: float, f2: float, allowed_error: float = 1.e-6) -> bool:
    return f1 <= f2 + allowed_error


def greater_or_equal(f1: float, f2: float,
                     allowed_error: float = 1.e-6) -> bool:
    return f1 >= f2 - allowed_error


def from_cartesian_to_cyl(pos: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    cyl_pos = np.array([np.sqrt(pos[:,0]**2+pos[:,1]**2),
                        np.arctan2(pos[:,1], pos[:,0]),
                        pos[:,2]]).transpose()
    return cyl_pos


def sel_coord(pos: Sequence[float], q: Sequence[float],
              threshold: float) -> Tuple[Sequence[float], Sequence[float]]:
    sel = q > threshold
    return pos[sel], q[sel]


def phi_mean_var(pos_phi: Sequence[float],
                 q: Sequence[float]) -> Tuple[float, float]:
    diff_sign = min(pos_phi) < 0 < max(pos_phi)
    if diff_sign & (np.abs(np.min(pos_phi))>np.pi/2):
        pos_phi[pos_phi<0] = np.pi + np.pi + pos_phi[pos_phi<0]
    mean_phi = np.average(pos_phi, weights=q)
    var_phi  = np.average((pos_phi-mean_phi)**2, weights=q)
    return mean_phi, var_phi


def calculate_phi_sigma_from_cart_coord(pos: Sequence[np.ndarray],
                                        q: Sequence[float], thr: float) -> float:
    pos_sel, q_sel = sel_coord(pos, q, thr)
    if len(pos_sel) != 0:
        pos_phi = np.arctan2(pos_sel[:,1], pos_sel[:,0])
        _, var_phi = phi_mean_var(pos_phi, q_sel)
        return np.sqrt(var_phi)
    else:
        return 1.e9


def find_SiPMs_over_threshold(df: pd.DataFrame,
                              threshold: float) -> pd.DataFrame:
    """
    Integrate the charge in time of each SiPM and select only those with
    total charge larger than threshold. This function is kept to be used
    with old test files.
    """
    tot_charges_df = df.groupby(['event_id','sensor_id'], as_index=False).charge.sum()
    return tot_charges_df[tot_charges_df.charge > threshold]


def find_closest_sipm(point: Tuple[float, float, float],
                      sipms: pd.DataFrame) -> pd.DataFrame:
   """
   Find the closest SiPM to a point, given a df of SiPMs.
   """
   sipm_xyz  = sipms[['X', 'Y', 'Z']].values
   distances = np.linalg.norm(np.subtract(sipm_xyz, point), axis=1)
   return sipms.iloc[np.argmin(distances)]



def divide_sipms_in_two_hemispheres(sns_ids: Sequence[int],
                                    sns_positions: Sequence[np.ndarray],
                                    sns_charges: Sequence[float],
                                    reference_pos: Tuple[float, float, float]) -> Tuple[Sequence[int],
                                                                                        Sequence[int],
                                                                                        Sequence[np.ndarray],
                                                                                        Sequence[np.ndarray],
                                                                                        Sequence[float],
                                                                                        Sequence[float]]:
    """
    Divide the SiPMs with charge between two hemispheres, using a given
    reference direction (reference_pos) as a discriminator.
    Return the lists of the ids, the charges and the positions of the
    SiPMs of the two groups.
    """
    sns_xy = np.array([sns_positions[:, 0], sns_positions[:, 1]]).transpose()
    ref_xy = np.array([reference_pos[0], reference_pos[1]])

    ### The scalar product is done only with the xy coordinates,
    ### to produce correct results also for points far from the FOV centre.

    scalar_prods = sns_xy.dot(ref_xy)

    mask1 = scalar_prods >  0
    mask2 = scalar_prods <= 0


    return (sns_ids      [mask1], sns_ids      [mask2],
            sns_positions[mask1], sns_positions[mask2],
            sns_charges  [mask1], sns_charges  [mask2])



def assign_sipms_to_gammas(sns_response: pd.DataFrame,
                           true_pos: Sequence[Tuple[float, float, float]],
                           DataSiPM_idx: pd.DataFrame) -> Tuple[Sequence[int],
                                                                Sequence[int],
                                                                Sequence[np.ndarray],
                                                                Sequence[np.ndarray],
                                                                Sequence[float],
                                                                Sequence[float]]:
    """
    Divide the SiPMs with charge between the two back-to-back gammas,
    or to one of the two if the other one hasn't interacted.
    Return the lists of the charges and the positions of the SiPMs of
    the two groups.
    DataSiPM_idx is assumed to be indexed on the sensor ids. If it is not,
    it is indexed inside the function.
    """
    if 'SensorID' in DataSiPM_idx.columns:
        DataSiPM_idx = DataSiPM_idx.set_index('SensorID')
    sipms           = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_ids         = sipms.index.astype('int64').values
    sns_pos         = np.array([sipms.X.values, sipms.Y.values, sipms.Z.values]).transpose()
    sns_closest_pos = [find_closest_sipm(pos, sipms)[['X', 'Y', 'Z']].values
                       for pos in true_pos]
    id1, id2, pos1, pos2, q1, q2 = divide_sipms_in_two_hemispheres(sns_ids, sns_pos,
                                                                   sns_response.charge,
                                                                   sns_closest_pos[0])

    if len(sns_closest_pos) == 1:
        return id1, [], pos1, [], q1, []
    else:
        return id1, id2, pos1, pos2, q1, q2


def initial_coord_first_daughter(particles: pd.DataFrame,
                                 mother_id: int) -> Tuple[Tuple[float, float, float],
                                                          float, str]:
    """
    Returns the position, time and volume of the initial vertex
    of the first daughter of a given particle.
    """
    daughters = particles[particles.mother_id == mother_id]
    if len(daughters):
        daughter = daughters.loc[daughters['initial_t'].idxmin()]
        ## Type protection needed in access for some reason.
        vtx_pos  = daughter[['initial_x',
                             'initial_y',
                             'initial_z']].values.astype('float32')
        init_vol = daughter.initial_volume
        return vtx_pos, daughter.initial_t, init_vol

    return [], float('inf'), None


def part_first_hit(hits: pd.DataFrame,
                   part_id: int) -> Tuple[Tuple[float, float, float], float, str]:
    """
    Returns the position, time and volume of the first hit of a given particle.
    """
    part_hits = hits[hits.particle_id == part_id]
    if len(part_hits):
        p_hit    = part_hits.loc[part_hits['time'].idxmin()]
        part_pos = p_hit[['x', 'y', 'z']].values
        return part_pos, p_hit.time, p_hit.label
    return [], float('inf'), None


def find_first_time_of_sensors(tof_response: pd.DataFrame,
                               sns_ids: Sequence[int],
                               n_pe: int = 1)-> Tuple[Sequence[int], float]:
    """
    This function looks for the time among all sensors for the first
    photoelectron detected.
    In case more than one photoelectron arrives at the same time,
    the sensor with minimum id is chosen.
    The positive value of the id of the sensor and the time of detection
    are returned.
    """
    tof = tof_response[tof_response.sensor_id.isin(sns_ids)]
    if tof.empty:
        raise WaveformEmptyTable("Tof dataframe is empty")

    if n_pe == 1:
        min_t   = tof.time.min()
        min_df  = tof[tof.time == min_t]
        min_ids = min_df.sensor_id.values
    else:
        first_times = tof.sort_values(by=['time']).iloc[0:n_pe]
        min_t       = first_times.time.mean()
        min_ids     = first_times.sensor_id.values

    return np.abs(min_ids), min_t


def find_hit_distances_from_true_pos(hits: pd.DataFrame,
                                     true_pos: Tuple[float, float, float]) -> Sequence[float]:
    """
    Calculates the distances of all the hits in the same hemisphere of a given point,
    from the given point.
    """
    positions       = hits[['x', 'y', 'z']].values
    scalar_products = positions.dot(true_pos)

    mask      = scalar_products >= 0
    diff      = np.subtract(positions[mask], true_pos)
    diff      = np.array(diff, dtype=np.float32)
    distances = np.linalg.norm(diff, axis=1)

    return distances


def find_b2b_gammas(particles: pd.DataFrame, source: bool = False)-> pd.DataFrame:
    """
    Returns a dataframe with the two almost back-to-back, almost 511 keV gammas.
    The case of two primary gammas (source == false) or a Na22 source are considered.
    """
    if source:
        gammas	  = particles[particles.particle_name == 'gamma']
        positrons = particles[(particles.particle_name == 'e+') &
                              (particles.initial_volume == 'NA22_SOURCE')]
        gammas    = gammas[(gammas.creator_proc == 'pos_annihil') &
                           (gammas.mother_id.isin(positrons.particle_id.values))]
    else:
        gammas = particles[particles.primary == True]

    return gammas

def calculate_initial_vertex(selected_part: pd.DataFrame,
                             hits: pd.DataFrame,
                             id1: int, id2: int)-> Tuple[Tuple[float, float, float],
                                                         Tuple[float, float, float],
                                                         float, float,
                                                         str, str]:
    """
    Looks for the first interaction of gammas.
    """

    ### Calculate the initial vertex.
    gamma_pos1, gamma_pos2 = [], []
    min_t1    , min_t2     = float('inf'), float('inf')
    vol1      , vol2       = None, None
    if len(selected_part[selected_part.mother_id == id1]) > 0:
        gamma_pos1, min_t1, vol1 = initial_coord_first_daughter(selected_part, id1)

    if len(selected_part[selected_part.mother_id == id2]) > 0:
        gamma_pos2, min_t2, vol2 = initial_coord_first_daughter(selected_part, id2)

    ### Calculate the minimum time among the hits of a given primary gamma,
    ### if any.
    if len(hits[hits.particle_id == id1]) > 0:
        g_pos1, g_min_t1, g_vol1 = part_first_hit(hits, id1)
        if g_min_t1 < min_t1:
            min_t1     = g_min_t1
            gamma_pos1 = g_pos1
            vol1       = g_vol1

    if len(hits[hits.particle_id == id2]) > 0:
        g_pos2, g_min_t2, g_vol2 = part_first_hit(hits, id2)
        if g_min_t2 < min_t2:
            min_t2     = g_min_t2
            gamma_pos2 = g_pos2
            vol2       = g_vol2

    gamma_pos1 = np.array(gamma_pos1, dtype=float)
    gamma_pos2 = np.array(gamma_pos2, dtype=float)

    return gamma_pos1, gamma_pos2, min_t1, min_t2, vol1, vol2


def find_first_interactions(particles: pd.DataFrame,
                            hits: pd.DataFrame,
                            vol: str = None,
                            source: bool = False)-> Tuple[Tuple[float, float, float],
                                                          Tuple[float, float, float],
                                                          float, float,
                                                          str, str]:
    """
    Returns the position, time and volume of the first interaction
    of each primary gammas.
    """
    gammas = find_b2b_gammas(particles, source)
    if len(gammas) != 2:
        print('Cannot find the two ~511 keV gammas for this event')
        return [], [], None, None, None, None

    id1 = gammas.particle_id.values[0]
    id2 = gammas.particle_id.values[1]

    ### select electrons
    if vol:
        particles = particles[(particles.initial_volume == vol) & (particles.final_volume == vol)]
    electrons  = particles[particles.particle_name == 'e-']
    daughter_e = electrons[electrons.mother_id.isin(gammas.particle_id.values)]

    ### Calculate the initial vertex.
    gamma_pos1, gamma_pos2, min_t1, min_t2, vol1, vol2 = calculate_initial_vertex(daughter_e, hits, id1, id2)
    if not len(gamma_pos1) or not len(gamma_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], None, None, None, None

    return gamma_pos1, gamma_pos2, min_t1, min_t2, vol1, vol2



def find_first_interactions_in_active(particles: pd.DataFrame,
                                      hits: pd.DataFrame,
                                      photo_range: float = 1.,
                                      petit: bool = False)-> Tuple[Tuple[float, float, float],
                                                                   Tuple[float, float, float],
                                                                   float, float,
                                                                   bool, bool]:
    """
    Returns the first interaction of each primary gammas in the active volume
    and distinguish if they are photoelectric-like or they aren't.
    """

    gamma_pos1, gamma_pos2, min_t1, min_t2, _, _ = find_first_interactions(particles, hits, 'ACTIVE', petit)
    if not len(gamma_pos1) or not len(gamma_pos2):
        return [], [], None, None, None, None

    ## find if the event is photoelectric-like
    distances1 = find_hit_distances_from_true_pos(hits, gamma_pos1)
    if max(distances1) > photo_range: ## hits at <1 mm distance are considered of the same point
        phot_like1 = False
    else:
        phot_like1 = True

    distances2 = find_hit_distances_from_true_pos(hits, gamma_pos2)
    if max(distances2) > photo_range: ## hits at <1 mm distance are considered of the same point
        phot_like2 = False
    else:
        phot_like2 = True

    return gamma_pos1, gamma_pos2, min_t1, min_t2, phot_like1, phot_like2


def reconstruct_coincidences(sns_response: pd.DataFrame,
                             charge_range: Tuple[float, float],
                             DataSiPM_idx: pd.DataFrame,
                             particles: pd.DataFrame,
                             hits: pd.DataFrame)-> Tuple[Sequence[Tuple[float, float, float]],
                                                   Sequence[Tuple[float, float, float]],
                                                   Sequence[float], Sequence[float],
                                                   Tuple[float, float, float],
                                                   Tuple[float, float, float],
                                                   float, float,
                                                   Sequence[int], Sequence[int]]:
    """
    Finds the SiPM with maximum charge. Divides the SiPMs in two groups,
    separated by the plane perpendicular to the line connecting this SiPM
    with the centre of the cylinder.
    The true position of the first gamma interaction in ACTIVE is also
    returned for each of the two primary gammas.
    The two SiPM groups are assigned to their correspondent true gamma by position.
    DataSiPM_idx is assumed to be indexed on the sensor ids. If it is not,
    it is indexed inside the function.
    """
    if 'SensorID' in DataSiPM_idx.columns:
        DataSiPM_idx = DataSiPM_idx.set_index('SensorID')

    max_sns  = sns_response.loc[sns_response['charge'].idxmax()].sensor_id
    max_sipm = DataSiPM_idx.loc[max_sns]
    max_pos  = max_sipm[['X', 'Y', 'Z']].values

    sipms         = DataSiPM_idx.loc[sns_response.sensor_id]
    sns_positions = sipms[['X', 'Y', 'Z']].values

    sns1, sns2, pos1, pos2, q1, q2 = divide_sipms_in_two_hemispheres(sipms.index.values,
                                                                     sns_positions,
                                                                     sns_response.charge.values,
                                                                     max_pos)

    tot_q1 = sum(q1)
    tot_q2 = sum(q2)

    sel1 = (tot_q1 > charge_range[0]) & (tot_q1 < charge_range[1])
    sel2 = (tot_q2 > charge_range[0]) & (tot_q2 < charge_range[1])
    if not sel1 or not sel2:
        return [], [], [], [], None, None, None, None, [], []

    true_pos1, true_pos2, true_t1, true_t2, _, _ = find_first_interactions_in_active(particles,
                                                                                     hits)

    if not len(true_pos1) or not len(true_pos2):
        print("Cannot find two true gamma interactions for this event")
        return [], [], [], [], None, None, None, None, [], []

    scalar_prod = true_pos1.dot(max_pos)
    if scalar_prod > 0:
        int_pos1 = pos1
        int_pos2 = pos2
        int_q1   = q1
        int_q2   = q2
        int_sns1 = sns1
        int_sns2 = sns2
    else:
        int_pos1 = pos2
        int_pos2 = pos1
        int_q1   = q2
        int_q2   = q1
        int_sns1 = sns2
        int_sns2 = sns1

    return int_pos1, int_pos2, int_q1, int_q2, true_pos1, true_pos2, true_t1, true_t2, int_sns1, int_sns2


def find_coincidence_timestamps(tof_response: pd.DataFrame,
                                sns1: Sequence[int],
                                sns2: Sequence[int],
                                n_pe: int = 1)-> Tuple[Tuple[int], Tuple[int], float, float]:
    """
    Finds the first time and the IDs of the sensors used to calculate it
    for each one of two sets of sensors, given a sensor response dataframe.
    """
    min1, time1 = find_first_time_of_sensors(tof_response, sns1, n_pe)
    min2, time2 = find_first_time_of_sensors(tof_response, sns2, n_pe)

    return min1, min2, time1, time2


def reconstruct_r_with_map(q: Sequence[float],
                           pos: Sequence[Tuple[float, float, float]],
                           Rmap: Map, thr_r: float = 0) -> float:
    """
    Calculates the r coordinate, given charges and positions of touched SiPM.
    """

    posr, qr = sel_coord(pos, q, thr_r)
    if len(posr) != 0:
        pos_phi = from_cartesian_to_cyl(posr)[:,1]
        _, var_phi = phi_mean_var(pos_phi, qr)
        r = Rmap(np.sqrt(var_phi)).value
    else:
        return 1.e9


def reconstruct_r_with_function(q: Sequence[float],
                                pos: Sequence[Tuple[float, float, float]],
                                a0: float, a1: float, a2: float, thr_r: float = 0) -> float:
    """
    Calculates the r coordinate, given charges and positions of touched SiPM.
    """

    def find_r(x):
        return a0 + a1*x + a2*x**2

    sig_phi = calculate_phi_sigma_from_cart_coord(pos, q, thr_r)
    if sig_phi < 1.e9:
        r = find_r(sig_phi)
    else:
        return 1.e9


def reconstruct_phi_z(q: Sequence[float],
                      pos: Sequence[Tuple[float, float, float]],
                      thr_phi: float = 0, thr_z: float = 0) -> Tuple[float, float]:
    """
    Calculates the phi and z coordinates, given charges and positions
    of touched SiPMs. Different thresholds can be used for each one of them.
    """

    ## Calculate phi
    posphi, qphi = sel_coord(pos, q, thr_phi)

    if len(qphi) != 0:
        reco_cart_pos = np.average(posphi, weights=qphi, axis=0)
        phi           = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
    else:
        return 1.e9, 1.e9

    ## Calculate z
    posz, qz = sel_coord(pos, q, thr_z)

    if len(qz) != 0:
        reco_cart_pos = np.average(posz, weights=qz, axis=0)
        z             = reco_cart_pos[2]
    else:
        return 1.e9, 1.e9

    return phi, z


def reconstruct_position(q: Sequence[float],
                         pos: Sequence[Tuple[float, float, float]],
                         Rmap: Map,
                         thr_r: float = 0, thr_phi: float = 0, thr_z: float = 0) -> Tuple[float,
                                                                                          float,
                                                                                          float]:
    """
    Calculates the r, phi and z coordinates, given charges and positions
    of touched SiPMs. Different thresholds can be used for each one of them.
    R is found using a previously calculated map.
    """
    r = reconstruct_r_with_map(pos, q, Rmap, thr_r)

    if r > 1.e8:
        return 1.e9, 1.e9, 1.e9

    phi, z = reconstruct_phi_z(q, pos, thr_phi, thr_z)

    return r, phi, z


def calculate_average_SiPM_pos(min_ids: Sequence[int],
                               DataSiPM_idx: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculates the geometrical average position of the SiPMs with IDs in the list.
    """
    sipm     = DataSiPM_idx.loc[np.abs(min_ids)]
    sipm_pos = sipm[['X', 'Y', 'Z']].values
    ave_pos  = np.average(sipm_pos, axis=0)

    return ave_pos
