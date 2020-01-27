import os

import pytest
import numpy as np

from . import fom_functions as fomf


@pytest.fixture(scope = 'module')
def phantom_true_image(ANTEADATADIR):
    return os.path.join(ANTEADATADIR, 'phantom_NEMAlike.npz')


def test_true_signal_crc_is_close_to_one(phantom_true_image):

    img_obj = np.load(phantom_true_image)
    img = img_obj['phantom']
    ### image characteristics
    max_intensity = 4
    #sig_sphere_r  = 11
    r             = 50
    #phi           = 4*np.pi/3
    bckg_sphere_r = 4
    phi0          = np.pi/6.
    phi_step      = np.pi/3.
    nphi          = 6
    x_size        = 180
    y_size        = 180
    xbins         = 180
    ybins         = 180

    sig_sphere_r = [4, 6.5, 8.5, 11]
    phi = [np.pi/3., 2.*np.pi/3., 3.*np.pi/3., 4*np.pi/3.]

    ### take one bin in z, in the centre  of the image
    img_slice = np.sum(img[:,:,89:90], axis=2)

    for i in range(0, len(phi)):
        crc = fomf.crc(img_slice, max_intensity, sig_sphere_r[i], r, phi[i], bckg_sphere_r,
                       phi0, phi_step, nphi, x_size, y_size, xbins, ybins)

        assert np.isclose(crc, 1, rtol=5e-02, atol=5e-02)
