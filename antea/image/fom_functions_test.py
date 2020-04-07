import os

import pytest
import numpy as np
import math

from . import fom_functions as fomf


@pytest.fixture(scope = 'module')
def phantom_true_img(ANTEADATADIR):
    img_file = os.path.join(ANTEADATADIR, 'phantom_NEMAlike.npz')
    img_obj  = np.load(img_file)
    img      = img_obj['phantom']

    return img

### image characteristics
sig_intensity  =   4
bckg_intensity =   1
r              =  50
bckg_sphere_r  =   4
phi0           = np.pi/6.
phi_step       = np.pi/3.
nphi           =   6
x_size         = 180
y_size         = 180
z_size         = 180
xbins          = 180
ybins          = 180
zbins          = 180

hot_sphere_r  = [4, 6.5, 8.5, 11]
hot_phi       = [np.pi/3., 2.*np.pi/3., 3.*np.pi/3., 4*np.pi/3.]
cold_sphere_r = [14, 18.5]
cold_phi      = [5*np.pi/3., 6*np.pi/3.]


def test_true_signal_crc_is_close_to_one(phantom_true_img):

    ### take one bin in z, in the centre of the image
    img_slice = np.sum(phantom_true_img[:,:,89:90], axis=2)

    for i in range(0, len(hot_phi)):
        crc, _ = fomf.crc_hot2d(img_slice, sig_intensity, bckg_intensity,
                                hot_sphere_r[i], r, hot_phi[i],
                                bckg_sphere_r, phi0, phi_step, nphi,
                                x_size, y_size, xbins, ybins)

        assert np.isclose(crc, 1, rtol=5e-02, atol=5e-02)

    ### take the 3d image
    for i in range(0, len(hot_phi)):
        crc, _ = fomf.crc_hot3d(phantom_true_img, sig_intensity, bckg_intensity,
                                hot_sphere_r[i], r, hot_phi[i],
                                bckg_sphere_r, phi0, phi_step, nphi,
                                x_size, y_size, z_size,
                                xbins, ybins, zbins)

        assert np.isclose(crc, 1, rtol=5e-02, atol=5e-02)


def test_true_background_crc_is_close_to_zero(phantom_true_img):

    ### take one bin in z, in the centre of the image
    img_slice = np.sum(phantom_true_img[:,:,89:90], axis=2)

    for i in range(0, len(cold_phi)):
        crc, _ = fomf.crc_cold2d(img_slice, cold_sphere_r[i],
                                 r, cold_phi[i], bckg_sphere_r,
                                 phi0, phi_step, nphi,
                                 x_size, y_size, xbins, ybins)

        assert np.isclose(crc, 1, rtol=1e-02, atol=1e-02)

    ### take the 3d image
    for i in range(0, len(cold_phi)):
        crc, _ = fomf.crc_cold3d(phantom_true_img, cold_sphere_r[i],
                                 r, cold_phi[i], bckg_sphere_r,
                                 phi0, phi_step, nphi,
                                 x_size, y_size, z_size, xbins, ybins, zbins)

        assert np.isclose(crc, 1, rtol=1e-02, atol=1e-02)


def test_signal_to_noise_ratio_is_infinite(phantom_true_img):

    ### take one bin in z, in the centre of the image
    img_slice = np.sum(phantom_true_img[:,:,89:90], axis=2)

    for i in range(0, len(hot_phi)):
        snr, _ = fomf.snr2d(img_slice, hot_sphere_r[i], r, hot_phi[i],
                            bckg_sphere_r, phi0, phi_step, nphi,
                            x_size, y_size, xbins, ybins)

        assert math.isinf(snr)

    ### take the 3d image
    for i in range(0, len(cold_phi)):
        snr, _ = fomf.snr3d(phantom_true_img, hot_sphere_r[i], r, hot_phi[i],
                            bckg_sphere_r, phi0, phi_step, nphi,
                            x_size, y_size, z_size, xbins, ybins, zbins)

        assert math.isinf(snr)
