.. antea documentation master file, created by
   sphinx-quickstart on Wed Jan 22 17:21:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
   :format: html

ANTEA: ANalysis Tool Environment for petAlo
===========================================
ANTEA contains functions for processing NEXUS-generated Monte Carlo events and
for full PET image reconstruction. It requires functionality from
`NEXT-IC <https://github.com/nextic/IC>`_.

Overview
--------
The basic procedure for generating simulated events and reconstructing a PET
image can be performed in three different ways:

1. :ref:`Full simulation: NEXUS --> Coincidence selection/reconstruction -> Image Reco <fullmc>`
2. :ref:`Fast simulation: NEXUS --> Fast MC --> Image Reco <fastmc>`
3. :ref:`Fast-fast simulation: Fast-fast MC --> Image Reco <fastfastmc>`

For image reconstruction, we use a :ref:`3D MLEM algorithm <imgreco>`.

Examples
--------
* `Image reconstruction with fast-fast MC <https://github.com/nextic/ANTEA/blob/master/docs/examples/petalo_reconstruction_fastfastmc_example.ipynb>`_
* `Creation of a NEMA phantom <https://github.com/nextic/ANTEA/blob/master/docs/examples/phantom_NEMA_example.ipynb>`_

Code reference
--------------
.. toctree::
   :maxdepth: 2

   src/code_mcsim
   src/code_reconstruction
   src/code_utils


Indices and tables:
-------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

How to update this documentation
--------------------------------
The source (written in `reStructuredText <https://en.wikipedia.org/wiki/ReStructuredText>`_)
for this documentation is located in the `ANTEA/docs` folder. The main source
file is called `index.rst`, and the other relevant source files are located
in the `src` subdirectory. Assuming the Python environment for IC is set up,
the documentation can be updated as follows.

1. Update the source files
##########################
Ensure that the local `ANTEA/docs` directory is up-to-date. Then update the
source files, noting the following:

a) **If no new .py file has been added** (documentation of new or updated
classes/functions has been added only to existing files that are already
included in the documentation), no updating of the source files should be required.

b) **If a new .py file has been created and fits within in the scope of the
existing source (.rst) files in the src/ directory**, it can be added as a
module to the relevant one. For example, for a new file `new_reco_module.py`
containing reconstruction functions, one can add to
`src/code_reconstruction.rst` the following::

   .. automodule:: antea.reco.new_reco_module
      :members:

c) **If a new .py file has been created that requires a new entry in the table of
contents (unrelated to the present documentation)**, a new `.rst` file can be
created and placed in the `src/` directory. An `automodule` can be added to this
new file as described in point b), and the file can be added to the `toctree` in
`index.rst`. For example, for a new documentation file `code_new.rst`, one can
update the table of contents as::

   .. toctree::
      :maxdepth: 2

      src/code_mcsim
      src/code_reconstruction
      src/code_utils
      src/code_new

d) Any documentation outside of python docstrings can be written in reStructuredText
directly into the `.rst` source files.

2. Build the updated documentation
##################################
This is done automatically by Read the Docs, but upon editing the documentation,
one may want to do a local build to see how it looks before committing/pushing
the changes. Run the following command to install sphinx::

   $ pip install sphinx

The extension for processing typehints must also be installed::

   $ pip install sphinx-autodoc-typehints

And also the Sphinx theme::

   $ pip install sphinx_rtd_theme

Now the updated documentation can be built by running, from within the `docs/`
directory::

   $ make html

All the updated files in `docs/` can then be committed and pushed to git.

**Note:** the documentation is built assuming IC is not installed, and certain
warning messages may appear in the build output because of this. These can be
ignored if they do not affect the final output.
