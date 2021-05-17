#
#  The fast MC generates pairs of interaction points based on
#  pre-determined matrices of true r, phi, and z coordinates vs. their
#  reconstructed error. Some of these matrices have two independent variables
#  It uses the true information coming from GEANT4 simulations
#

import numpy  as np
import pandas as pd

from typing import Sequence, Tuple

from invisible_cities.core import system_of_units as units

from antea.mcsim.errmat   import errmat
from antea.mcsim.errmat3d import errmat3d
import antea.reco.reco_functions as rf


def compute_error_mat_2d(x: Sequence[float], diff_mat: Sequence[float], bins: Tuple[float, float],
                         ranges: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[Sequence[float],
                                                                                           float,
                                                                                           float,
                                                                                           float,
                                                                                           float,
                                                                                           float]:
    """
    Compute the 2d histogram and its edges for a variable and its errors to build the error matrices.
    """
    h, edges = np.histogramdd((x, diff_mat), bins=bins, range=ranges)
    eff      = np.array([1])
    xedges   = edges[0]
    yedges   = edges[1]
    xmin     = xedges[0]; xmin = np.array(xmin)
    ymin     = yedges[0]; ymin = np.array(ymin)
    dx       = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
    dy       = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
    return h, eff, xmin, ymin, dx, dy


def compute_error_mat_3d(x: Sequence[float], y: Sequence[float], diff_mat: Sequence[float], bins: Tuple[float, float, float],
                         ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> Tuple[Sequence[float],
                                                                                                                float,
                                                                                                                float,
                                                                                                                float,
                                                                                                                float,
                                                                                                                float,
                                                                                                                float,
                                                                                                                float]:
    """
    Compute the 3d histogram and its edges for a variable and its errors to build the error matrices.
    """
    h, edges = np.histogramdd((x, y, diff_mat), bins=bins, range=ranges)
    eff      = np.array([1])
    xedges   = edges[0]
    yedges   = edges[1]
    zedges   = edges[2]
    xmin     = xedges[0]; xmin = np.array(xmin)
    ymin     = yedges[0]; ymin = np.array(ymin)
    zmin     = zedges[0]; zmin = np.array(zmin)
    dx       = xedges[1:]-xedges[:-1]; dx = np.array(dx[0])
    dy       = yedges[1:]-yedges[:-1]; dy = np.array(dy[0])
    dz       = zedges[1:]-zedges[:-1]; dz = np.array(dz[0])
    return h, eff, xmin, ymin, zmin, dx, dy, dz


def get_reco_interaction(r: float, phi: float, z: float, t: float,
                         errmat_r: errmat, errmat_phi: errmat3d, errmat_z: errmat3d, errmat_t: errmat):
    """
    Extract the spatial coordinates and time for one interaction, using error matrices.
    """
    reco_r   = errmat_r  .get_random_error(r)
    reco_phi = errmat_phi.get_random_error(phi, r)
    reco_z   = errmat_z  .get_random_error(z, r)
    reco_t   = errmat_t  .get_random_error(t)

    return reco_r, reco_phi, reco_z, reco_t

def simulate_reco_event(evt_id: int, hits: pd.DataFrame, particles: pd.DataFrame,
                        errmat_p_r: errmat, errmat_p_phi: errmat3d, errmat_p_z: errmat3d,
                        errmat_p_t: errmat, errmat_c_r: errmat, errmat_c_phi: errmat3d,
                        errmat_c_z: errmat3d, errmat_c_t: errmat,
                        true_e_threshold: float = 0., photo_range: float = 1.,
                        only_phot: bool = False) -> pd.DataFrame:
    """
    Simulate the reconstructed coordinates for 1 coincidence from true GEANT4 dataframes.
    """

    evt_parts = particles[particles.event_id == evt_id]
    evt_hits  = hits     [hits     .event_id == evt_id]
    energy    = evt_hits.energy.sum()
    if energy < true_e_threshold:
        return None

    pos1, pos2, t1, t2, phot1, phot2 = rf.find_first_interactions_in_active(evt_parts, evt_hits, photo_range)

    no_reco_positions = len(pos1) == 0 or len(pos2) == 0
    no_phot_interactions = not phot1 or not phot2
    if no_reco_positions or (only_phot and no_phot_interactions):
        return None

    t1 = t1 / units.ps
    t2 = t2 / units.ps

    # Transform in cylindrical coordinates
    (r1, phi1, z1), (r2, phi2, z2) = rf.from_cartesian_to_cyl(np.array([pos1, pos2]))

    # Get all errors.
    if phot1:
        er1, ephi1, ez1, et1 = get_reco_interaction(r1, phi1, z1, t1, errmat_p_r, errmat_p_phi, errmat_p_z, errmat_p_t)
    else:
        er1, ephi1, ez1, et1 = get_reco_interaction(r1, phi1, z1, t1, errmat_c_r, errmat_c_phi, errmat_c_z, errmat_c_t)

    if phot2:
        er2, ephi2, ez2, et2 = get_reco_interaction(r2, phi2, z2, t2, errmat_p_r, errmat_p_phi, errmat_p_z, errmat_p_t)
    else:
        er2, ephi2, ez2, et2 = get_reco_interaction(r2, phi2, z2, t2, errmat_c_r, errmat_c_phi, errmat_c_z, errmat_c_t)

    if er1 == None or ephi1 == None or ez1 == None or et1 == None or er2 == None or ephi2 == None or ez2 == None or et2 == None:
        return None


    # Compute reconstructed quantities.
    r1_reco = r1 - er1
    r2_reco = r2 - er2
    phi1_reco = phi1 - ephi1
    phi2_reco = phi2 - ephi2
    z1_reco = z1 - ez1
    z2_reco = z2 - ez2
    t1_reco = t1 - et1
    t2_reco = t2 - et2

    events = pd.DataFrame({'event_id':    float(evt_id),
                           'true_energy': energy,
                           'true_r1':     r1,
                           'true_phi1':   phi1,
                           'true_z1':     z1,
                           'true_t1':     t1,
                           'true_r2':     r2,
                           'true_phi2':   phi2,
                           'true_z2':     z2,
                           'true_t2':     t2,
                           'phot_like1':  float(phot1),
                           'phot_like2':  float(phot2),
                           'reco_r1':     r1_reco,
                           'reco_phi1':   phi1_reco,
                           'reco_z1':     z1_reco,
                           'reco_t1':     t1_reco,
                           'reco_r2':     r2_reco,
                           'reco_phi2':   phi2_reco,
                           'reco_z2':     z2_reco,
                           'reco_t2':     t2_reco}, index=[0])
    return events
