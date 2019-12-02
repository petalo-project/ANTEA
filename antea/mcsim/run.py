# Run a fast fast MC
import antea.fastfastmc.fastfastmc as ffmc

from errmat import errmat
from phantom import phantom

Nevts = 1000

# Construct the phantom object.
phtm = phantom("/Users/jrenner/local/jerenner/ANTEA/antea/fastfastmc/phantom/phantom_NEMAlike.npz")

# Construct the error matrix objects.
errmat_r = errmat('/Users/jrenner/local/jerenner/ANTEA/antea/fastfastmc/errmat/errmat_r.npz')
errmat_phi = errmat('/Users/jrenner/local/jerenner/ANTEA/antea/fastfastmc/errmat/errmat_phi.npz')
errmat_z = errmat('/Users/jrenner/local/jerenner/ANTEA/antea/fastfastmc/errmat/errmat_z.npz')

# Run the simulation.
events = ffmc.run_fastfastmc(Nevts, phtm, errmat_r, errmat_phi, errmat_z)

# Save the results to file.
events.to_hdf("sim.h5","fastfastsim")
