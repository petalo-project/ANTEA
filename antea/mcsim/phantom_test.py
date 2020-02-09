import os
import pytest
import struct

import numpy                 as np
import antea.mcsim.phantom   as ph
import hypothesis.strategies as st

from hypothesis  import given

@given(st.integers(min_value=1, max_value=8))
def test_create_sphere(rsphere):
    """
    Tests the create_sphere function.
    """
    vsphere = ph.create_sphere(rsphere)

    # Volume should be greater than theoretical volume for r-1.
    assert (np.sum(vsphere) > 4*np.pi*(rsphere-1)**3/3)

@given(st.integers(min_value=1, max_value=8), st.integers(min_value=1, max_value=10))
def test_create_cylinder(rcylinder, hhcylinder):
    """
    Tests the create_cylinder function.
    """
    vcylinder = ph.create_cylinder(rcylinder, hhcylinder)

    # Volume should be greater than theoretical volume for r-1.
    assert (np.sum(vcylinder) > np.pi*(rcylinder-1)**2*(2*hhcylinder))

@given(st.integers(min_value=1, max_value=8))
def test_phantom_add_to_vol(rsphere):
    """
    Tests the add_to_vol function of the phantom class.
    """
    # Create and add the volume.
    phtm    = ph.phantom(2*rsphere+1,2*rsphere+1,2*rsphere+1)
    vadd    = ph.create_sphere(rsphere)
    phtm.add_to_vol(vadd,0,0,0)

    # Ensure the entire volume was added to the (previously empty) phantom.
    assert (np.sum(phtm.get_volume()) == np.sum(vadd))

@pytest.mark.slow
def test_phantom_save_cdist_binary(output_tmpdir, ANTEADATADIR):
    """
    Tests the save_cdist_binary function of the phantom class.
    """
    PATH_IN  = os.path.join(ANTEADATADIR,  'phantom_NEMAlike.npz')
    PATH_OUT = os.path.join(output_tmpdir, 'phantom_NEMAlike.dat')

    # Read in the phantom in the test data directory, and save it in binary format.
    nema = ph.phantom(phantom_file=PATH_IN)
    nema.save_cdist_binary(PATH_OUT)

    # Perform some checks on the saved file.
    f2 = open(PATH_OUT,'rb')

    # Read and check the header.
    Nx = f2.read(4)
    Ny = f2.read(4)
    Nz = f2.read(4)
    Lx = f2.read(4)
    Ly = f2.read(4)
    Lz = f2.read(4)
    assert (struct.unpack('i',Nx)[0] == 180)
    assert (struct.unpack('i',Ny)[0] == 180)
    assert (struct.unpack('i',Nz)[0] == 180)
    assert (struct.unpack('f',Lx)[0] == 180.0)
    assert (struct.unpack('f',Ly)[0] == 180.0)
    assert (struct.unpack('f',Lz)[0] == 180.0)

    # Read the remainder of the values.
    for ii in range(5832000): f2.read(4)
    f2.close()
