#
# The fast fast MC generates pairs of interaction points based on
#  pre-determined matrices of true r, phi, and z coordinates vs. their
#  reconstructed error.
#
import numpy  as np
import pandas as pd

from . errmat import errmat
from . phantom import phantom

def run_fastfastmc(Nevts: int, phtm: phantom, errmat_r: errmat, errmat_phi: errmat,
                    errmat_z: errmat, rmin: float = 380.0, zmin: float = -450.0,
                    zmax: float = 450.0, coslim: float = 0.309) -> pd.DataFrame:
    """
    Runs the fast fast MC, simulating coincident events in a PETALO geometry
    subject to the specified restrictions.

    :param Nevts: the number of events to generate
    :type Nevts: int
    :param phtm: the phantom to simulate
    :type phtm: class antea.mcsim.phantom
    :param errmat_r: the r-error matrix
    :type errmat_r: class antea.mcsim.errmat
    :param errmat_phi: the phi-error matrix
    :type errmat_phi: class antea.mcsim.errmat
    :param errmat_z: the z-error matrix
    :type errmat_z: class antea.mcsim.errmat
    :param rmin: the radius at which the LXe volume begins
    :type rmin: float
    :param zmin: minimum z-coordinate simulated (see note below)
    :type zmin: float
    :param zmax: maximum z-coordinate simulated (see note below)
    :type zmax: float
    :param coslim: limit on the cosine of the opening angle of emitted gammas
     (see note below)
    :type coslim: float
    :rtype: Pandas dataframe containing the following information for each
     event: 'event_id', 'true_r1', 'true_phi1', 'true_z1', 'true_t1',
     'true_r2', 'true_phi2', 'true_z2', 'true_t2', 'reco_r1', 'reco_phi1',
     'reco_z1', 'reco_r2', 'reco_phi2', 'reco_z2'

    **Note** that there are two ways to restrict the z-extent of the generated coincidences:
        1. direct restriction of the opening angle: this is done by specifying
            coslim > 0. In this case the gammas emitted from each point will be
            restricted to opening angle with cosine from (-coslim, coslim)
        2. restriction of the z-extent of the gamma interactions: this is done
            by specifying coslim < 0 and meaningful values for rmin, zmin, and
            zmax. In this case gammas will be generated so that they interact
            only within (zmin, zmax).
    """

    # Pick a random number for the location of the emission point.
    ievts = np.random.choice(len(phtm.cdist),Nevts,p=phtm.cdist)

    # Compute the cosines and sines of the axial angle.
    phis = np.random.uniform(size=Nevts)*2*np.pi
    cosines_phi = np.cos(phis)
    sines_phi = np.sin(phis)

    # Set up empty arrays to contain the event information.
    a_event_ids = []
    a_true_r1 = []; a_true_phi1 = []; a_true_z1 = []; a_true_t1 = []
    a_true_r2 = []; a_true_phi2 = []; a_true_z2 = []; a_true_t2 = []
    a_reco_r1 = []; a_reco_phi1 = []; a_reco_z1 = []; a_reco_t1 = []
    a_reco_r2 = []; a_reco_phi2 = []; a_reco_z2 = []; a_reco_t2 = []

    # Simulate the events.
    for ievt,(ii,cphi,sphi,phi) in enumerate(zip(ievts,cosines_phi,sines_phi,phis)):

        # Determine the point from which the gammas are emitted.
        nnx = int(ii / phtm.NyNz)
        nny = int(ii/phtm.Nz) % phtm.Ny
        nnz = int(ii) % phtm.Nz

        # Convert to world coordinates:
        #  By convention (x, y, z) = (0.0, 0.0, 0.0) cooresponds to the center
        #  of the modeled volume.
        xpt = phtm.Lx*(1.0*nnx/phtm.Nx - 0.5)
        ypt = phtm.Ly*(1.0*nny/phtm.Ny - 0.5)
        zpt = phtm.Lz*(1.0*nnz/phtm.Nz - 0.5)

        # Compute the azimuthal angle.
        if(coslim < 0):
            clim_high = (zmax - zpt)/(rmin**2 + (zmax-zpt)**2)**0.5
            clim_low = (zmin - zpt)/(rmin**2 + (zmin-zpt)**2)**0.5
            clim = min(abs(clim_low),abs(clim_high))
            cth = np.random.uniform(-clim,clim)
        else:
            cth = np.random.uniform(-coslim,coslim)
        sth = (1-cth**2)**0.5

        # Get 2 random radii.
        rc1 = errmat_r.get_random_coord()
        rc2 = errmat_r.get_random_coord()

        # Determine the radii extending from the emission point for the corresponding random radii
        #  (which extend from the origin).  Note that these radii are in the x-y plane.
        rpt_sq = xpt**2 + ypt**2 + zpt**2
        b1 = 2*(xpt*cphi*sth + ypt*sphi*sth)
        c1 = rpt_sq - rc1**2 - zpt**2
        rp1 = np.max(np.roots([sth**2, b1, c1]))*np.abs(sth)

        b2 = -b1
        c2 = rpt_sq - rc2**2 - zpt**2
        rp2 = np.max(np.roots([sth**2, b2, c2]))*np.abs(sth)

        # Compute the full radii from the emission point to each interaction point (including z).
        rstar1 = rp1/np.abs(sth)
        rstar2 = rp2/np.abs(sth)

        # Compute the two "interaction points".
        x1 = xpt + rstar1*cphi*sth
        x2 = xpt - rstar2*cphi*sth
        y1 = ypt + rstar1*sphi*sth
        y2 = ypt - rstar2*sphi*sth
        z1 = zpt + rstar1*cth
        z2 = zpt - rstar2*cth

        # Convert to cylindrical coordinates.
        r1 = (x1**2 + y1**2)**0.5  # This should be equal to rc1
        phi1 = np.arctan2(y1,x1)
        r2 = (x2**2 + y2**2)**0.5
        phi2 = np.arctan2(y2,x2)

        # Get all errors.
        er1 = errmat_r.get_random_error(r1)
        er2 = errmat_r.get_random_error(r2)
        ephi1 = errmat_phi.get_random_error(phi1)
        ephi2 = errmat_phi.get_random_error(phi2)
        ez1 = errmat_z.get_random_error(z1)
        ez2 = errmat_z.get_random_error(z2)

        # Compute reconstructed quantities.
        r1_reco = r1 - er1
        r2_reco = r2 - er2
        phi1_reco = phi1 - ephi1
        phi2_reco = phi2 - ephi2
        z1_reco = z1 - ez1
        z2_reco = z2 - ez2

        # Compute (in ns) the TOF.
        tof = 1.0e9*(((x2-xpt)**2 + (y2-ypt)**2 + (z2-zpt)**2)**0.5 - ((x1-xpt)**2 + (y1-ypt)**2 + (z1-zpt)**2)**0.5)/3.0e11

        # Add the values for this event to the arrays.
        a_event_ids.append(ievt)

        a_true_r1.append(r1)
        a_true_phi1.append(phi1)
        a_true_t1.append(0.)
        a_true_z1.append(z1)
        a_true_r2.append(r2)
        a_true_phi2.append(phi2)
        a_true_z2.append(z2)
        a_true_t2.append(tof)

        a_reco_r1.append(r1_reco)
        a_reco_r2.append(r2_reco)
        a_reco_phi1.append(phi1_reco)
        a_reco_phi2.append(phi2_reco)
        a_reco_z1.append(z1_reco)
        a_reco_z2.append(z2_reco)

        if(ievt % (Nevts/10) == 0):
            print("Done {} events".format(ievt))

    events = pd.DataFrame({'event_id':  a_event_ids,
                           'true_r1':   a_true_r1,
                           'true_phi1': a_true_phi1,
                           'true_z1':   a_true_z1,
                           'true_t1':   a_true_t1,
                           'true_r2':   a_true_r2,
                           'true_phi2': a_true_phi2,
                           'true_z2':   a_true_z2,
                           'true_t2':   a_true_t2,
                           'reco_r1':   a_reco_r1,
                           'reco_phi1': a_reco_phi1,
                           'reco_z1':   a_reco_z1,
                           'reco_r2':   a_reco_r2,
                           'reco_phi2': a_reco_phi2,
                           'reco_z2':   a_reco_z2})
    return events
