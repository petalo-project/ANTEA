Reconstruction
********************

Coincidences reconstruction
===========================

To reconstruct a pair of coincidences for each MC event, first the SiPMs that have recorded charge above a certain threshold are
divided in two halves, identifying the SiPM with maximum charge and considering it at the centre of the first interaction.
A plane perpendicular to the line connecting the position of that SiPM with the geometrical centre of the system, and passing
through the centre of the system, divides the sensors in two halves, each one corresponding to one gamma interaction.
The true position and time of the interactions are also extracted to help the study of the performance of the algorithms.

.. automodule:: antea.reco.reco_functions
   :members:

MC truth selection functions
============================
For some studies it is useful to identify the events where one of the gammas has interacted via photoelectric effect only,
using the information of the true interactions happening in the MC.

.. automodule:: antea.reco.mctrue_functions
   :members:
