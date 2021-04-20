MC simulations
**************

.. _fullmc:

Full MC
=======
In the full MC, NEXUS is run in full simulation mode, which propagates scintillation
photons produced by the 511 keV gamma rays to the SiPMs so that the positions
of the interactions can be reconstructed. Event selection, position reconstruction,
and the final PET image reconstruction are performed using functions from
antea.reco.

.. _fastmc:

Fast MC
=======
In the fast MC, NEXUS is run without propagation of the photons, yielding only the
true interaction positions of the simulated 511 keV gamma rays. With the "fast MC",
reconstructed interaction positions are then determined randomly using error
matrices produced in an independent NEXUS full simulation for the same
geometric configuration. The PET image can then be reconstructed using functions
from antea.reco.

.. automodule:: antea.mcsim.fastmc3d
   :members:

.. _fastfastmc:

Fast-fast MC
============
With the "fast-fast MC", the simulation of true interaction positions of
coincident events is performed outside of NEXUS, and the reconstructed interaction
positions are simulated using error matrices generated in an independent NEXUS
full simulation with the appropriate geometric configuration. Only coincident
events are simulated in which the full 511 keV was deposited by each of the two
gammas, and the error matrices do not account for correlations between
coordinates, and therefore this is the least physically accurate simulation
method. However it is also the fastest and may be used to get a rough idea of
image reconstruction quality with varying numbers of coincident events. The PET
image can be reconstructed immediately from the quantities produced by the
fast-fast MC using functions from antea.reco.

.. automodule:: antea.mcsim.fastfastmc
   :members:

MC utilities
============

.. automodule:: antea.mcsim.errmat
   :members:

.. automodule:: antea.mcsim.errmat3d
   :members:

Functions for creating basic phantoms
-------------------------------------
.. automodule:: antea.mcsim.phantom
   :members:
