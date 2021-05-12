
import os
import pandas as pd
import numpy  as np

from .. reco    import reco_functions as rf
from .. io      import mc_io          as mcio
from .          import fastmc3d       as fmc
from . errmat   import errmat
from . errmat3d import errmat3d


def test_compute_error_mat_2d():

    r      = np.random.uniform(low=380, high=410, size=(50,))
    err    = np.random.uniform(low=-1,  high=1,   size=(50,))
    bins   = (30, 100)
    ranges = (380, 410), (-1, 1)
    h, eff, xmin, ymin, dx, dy = fmc.compute_error_mat_2d(r, err, bins, ranges)

    assert h.shape == bins
    assert eff     == 1
    assert xmin    == ranges[0][0]
    assert ymin    == ranges[1][0]
    assert np.isclose(dx, (ranges[0][1] - ranges[0][0])/bins[0])
    assert np.isclose(dy, (ranges[1][1] - ranges[1][0])/bins[1])


def test_compute_error_mat_3d():

    r      = np.random.uniform(low=380,   high=410,  size=(50,))
    phi    = np.random.uniform(low=-3.15, high=3.15, size=(50,))
    err    = np.random.uniform(low=-1,    high=1,    size=(50,))
    bins   = (30, 100, 100)
    ranges = (380, 410), (-3.15, 3.15), (-1, 1)
    h, eff, xmin, ymin, zmin, dx, dy, dz = fmc.compute_error_mat_3d(r, phi, err, bins, ranges)

    assert h.shape == bins
    assert eff     == 1
    assert xmin    == ranges[0][0]
    assert ymin    == ranges[1][0]
    assert zmin    == ranges[2][0]
    assert np.isclose(dx, (ranges[0][1] - ranges[0][0])/bins[0])
    assert np.isclose(dy, (ranges[1][1] - ranges[1][0])/bins[1])
    assert np.isclose(dz, (ranges[2][1] - ranges[2][0])/bins[2])


def test_simulate_reco_event(ANTEADATADIR):

    PATH_IN = os.path.join(ANTEADATADIR, 'full_body_fastsim_test.h5')

    # Construct the error matrix objects.
    err_r_phot_file     = os.path.join(ANTEADATADIR, 'errmat_test_r_phot_like.npz')
    err_r_compt_file    = os.path.join(ANTEADATADIR, 'errmat_test_r_compt_like.npz')
    err_phi_phot_file   = os.path.join(ANTEADATADIR, 'errmat_test_phi_phot_like.npz')
    err_phi_compt_file  = os.path.join(ANTEADATADIR, 'errmat_test_phi_compt_like.npz')
    err_z_phot_file     = os.path.join(ANTEADATADIR, 'errmat_test_z_phot_like.npz')
    err_z_compt_file    = os.path.join(ANTEADATADIR, 'errmat_test_z_compt_like.npz')
    err_t_th_phot_file  = os.path.join(ANTEADATADIR, 'errmat_test_t_phot_like.npz')
    err_t_th_compt_file = os.path.join(ANTEADATADIR, 'errmat_test_t_compt_like.npz')

    errmat_r_phot     = errmat  (err_r_phot_file    )
    errmat_r_compt    = errmat  (err_r_compt_file   )
    errmat_phi_phot   = errmat3d(err_phi_phot_file  )
    errmat_phi_compt  = errmat3d(err_phi_compt_file )
    errmat_z_phot     = errmat3d(err_z_phot_file    )
    errmat_z_compt    = errmat3d(err_z_compt_file   )
    errmat_t_th_phot  = errmat  (err_t_th_phot_file )
    errmat_t_th_compt = errmat  (err_t_th_compt_file)

    particles = mcio.load_mcparticles(PATH_IN)
    hits      = mcio.load_mchits     (PATH_IN)
    events    = particles.event_id.unique()

    cols = ['event_id', 'true_energy',
            'true_r1', 'true_phi1', 'true_z1', 'true_t1',
            'true_r2', 'true_phi2', 'true_z2', 'true_t2',
            'phot_like1', 'phot_like2',
            'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
            'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2']
    reco0 = pd.DataFrame(columns=cols)
    reco1 = pd.DataFrame(columns=cols)

    energy_threshold = 1.

    for evt in events:

        evt_df0 = fmc.simulate_reco_event(evt, hits, particles,
                                          errmat_r_phot,  errmat_phi_phot,  errmat_z_phot,  errmat_t_th_phot,
                                          errmat_r_compt, errmat_phi_compt, errmat_z_compt, errmat_t_th_compt,
                                          energy_threshold, photo_range=1., only_phot=True)
        evt_df1 = fmc.simulate_reco_event(evt, hits, particles,
                                          errmat_r_phot,  errmat_phi_phot,  errmat_z_phot,  errmat_t_th_phot,
                                          errmat_r_compt, errmat_phi_compt, errmat_z_compt, errmat_t_th_compt,
                                          energy_threshold, photo_range=1., only_phot=False)

        reco0 = pd.concat([reco0, evt_df0])
        reco1 = pd.concat([reco1, evt_df1])

        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        energy    = evt_hits.energy.sum()

        if energy < energy_threshold:
            assert evt_df0 is None

        pos1, pos2, t1, t2, phot1, phot2 = rf.find_first_interactions_in_active(evt_parts, evt_hits, photo_range=1.)
        if not phot1 or not phot2:
            assert evt_df0 is None

    sel = (reco1.phot_like1.values==1) & (reco1.phot_like2.values==1)
    assert reco0.event_id.values == reco1[sel].event_id.values
