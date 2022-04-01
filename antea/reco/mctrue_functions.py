import numpy  as np
import pandas as pd

from typing     import Sequence, Tuple

from antea.reco import reco_functions   as rf


def find_hits_of_given_particles(p_ids: Sequence[int], hits: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the hits belonging to given particles.
    """
    return hits[hits.particle_id.isin(p_ids)]


def select_photoelectric(evt_parts: pd.DataFrame, evt_hits: pd.DataFrame, noncoll_gammas: bool = False) -> Tuple[bool, Sequence[Tuple[float, float, float]]]:
    """
    Select only the events where one or two photoelectric events occur, and nothing else.
    """
    sel_volume   = (evt_parts.initial_volume == 'ACTIVE') & (evt_parts.final_volume == 'ACTIVE')
    sel_name     =  evt_parts.particle_name == 'e-'
    sel_vol_name = evt_parts[sel_volume & sel_name]
    ids          = sel_vol_name.particle_id.values

    sel_hits   = find_hits_of_given_particles(ids, evt_hits)
    energies   = sel_hits.groupby(['particle_id'])[['energy']].sum()
    energies   = energies.reset_index()

    if noncoll_gammas:
        energy_sel = energies[(energies.energy >= 0.473) & (energies.energy <= 0.513)]
        gammas     = evt_parts[evt_parts.particle_name == 'gamma']
    else:
        energy_sel = energies[rf.greater_or_equal(energies.energy, 0.476443, allowed_error=1.e-6)]
        gammas     = evt_parts[evt_parts.primary == True]

    sel_vol_name_e = sel_vol_name  [sel_vol_name  .particle_id.isin(energy_sel.particle_id)]
    sel_all        = sel_vol_name_e[sel_vol_name_e.mother_id  .isin(gammas.particle_id.values)]
    if len(sel_all) == 0:
        return (False, [])

    ### Once the event has passed the selection, let's calculate the true position(s)
    ids      = sel_all.particle_id.values
    sel_hits = find_hits_of_given_particles(ids, evt_hits)

    sel_hits = sel_hits.groupby(['particle_id'])
    true_pos = []
    for _, df in sel_hits:
        hit_positions = np.array([df.x.values, df.y.values, df.z.values]).transpose()
        true_pos.append(np.average(hit_positions, axis=0, weights=df.energy))

    if not noncoll_gammas:
        ### Reject events where the two gammas have interacted in the same hemisphere.
        if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511):
           return (False, [])

    return (True, true_pos)
