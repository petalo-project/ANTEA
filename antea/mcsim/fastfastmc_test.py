import os

from  .            import fastfastmc as ffmc
from  . errmat     import errmat
from  . phantom    import phantom

def test_run_fastfastmc(ANTEADATADIR):
    """
    Test run of the fast fast MC.
    """
    Nevts = 1000

    # Construct the phantom object.
    PATH_PHANTOM = os.path.join(ANTEADATADIR, 'phantom_NEMAlike.npz')
    phtm = phantom(phantom_file=PATH_PHANTOM)

    # Construct the error matrix objects.
    PATH_ERRMAT_R = os.path.join(ANTEADATADIR, 'errmat_r.npz')
    errmat_r = errmat(PATH_ERRMAT_R)
    PATH_ERRMAT_PHI = os.path.join(ANTEADATADIR, 'errmat_phi.npz')
    errmat_phi = errmat(PATH_ERRMAT_PHI)
    PATH_ERRMAT_Z = os.path.join(ANTEADATADIR, 'errmat_z.npz')
    errmat_z = errmat(PATH_ERRMAT_Z)

    # Run the simulation.
    events = ffmc.run_fastfastmc(Nevts, phtm, errmat_r, errmat_phi, errmat_z)

    # Ensure the number of events simulated is Nevts.
    if(len(events)):
        assert(len(events.event_id) == 1000)
