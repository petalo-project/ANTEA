import os

import pytest
import numpy as np

from . import fom_functions as fomf


@pytest.fixture(scope = 'module')
def phantom_true_img(ANTEADATADIR):
    img_file = os.path.join(ANTEADATADIR, 'phantom_NEMAlike.npz')
    img_obj  = np.load(img_file)
    img      = img_obj['phantom']

    return img

### image characteristics
max_intensity = 4
r             = 50
bckg_sphere_r = 4
phi0          = np.pi/6.
phi_step      = np.pi/3.
nphi          = 6
x_size        = 180
y_size        = 180
xbins         = 180
ybins         = 180

hot_sphere_r = [4, 6.5, 8.5, 11]
hot_phi      = [np.pi/3., 2.*np.pi/3., 3.*np.pi/3., 4*np.pi/3.]
cold_sphere_r = [14, 18.5, 22.]
cold_phi      = [5*np.pi/3.,  6*np.pi/3., 0.]


def test_true_signal_crc_is_close_to_one(phantom_true_img):

    ### take one bin in z, in the centre  of the image
    img_slice = np.sum(phantom_true_img[:,:,89:90], axis=2)

    for i in range(0, len(hot_phi)):
        crc = fomf.crc2d(img_slice, max_intensity, hot_sphere_r[i], r, hot_phi[i],
                        bckg_sphere_r, phi0, phi_step, nphi, x_size, y_size,
                        xbins, ybins)

        assert np.isclose(crc, 1, rtol=5e-02, atol=5e-02)


def test_true_background_crc_is_close_to_zero(phantom_true_img):

    ### take one bin in z, in the centre  of the image
    img_slice = np.sum(phantom_true_img[:,:,89:90], axis=2)

    for i in range(0, len(cold_phi)):
        crc = fomf.crc2d(img_slice, max_intensity, cold_sphere_r[i], r, cold_phi[i],
                        bckg_sphere_r, phi0, phi_step, nphi, x_size, y_size,
                        xbins, ybins)

        assert np.isclose(crc, 0, rtol=8e-02, atol=8e-02)
