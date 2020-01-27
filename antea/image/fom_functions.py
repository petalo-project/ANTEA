import numpy as np

"""
Functions that provide figure of merits for image reconstruction.
The image is assumed to be a phantom with spheres of different radii on
a background.
"""


def average_in_sphere(img2d : np.ndarray,
                      sphere_r : float, r : float, phi : float,
                      x_size : float, y_size : float, xbins : int, ybins : int) -> float:
    """
    Calculates the average density in a sphere of radius sphere_r, centred in (r, phi).
    x_size, y_size: size of the image (in length unit).
    xbins, ybins: number of bins  in x and y.
    """

    x_bin_width = x_size / xbins
    y_bin_width = y_size / ybins
    cx = r*np.cos(phi)
    cy = r*np.sin(phi)

    sphere_bins = []
    for i in range(xbins):
        for j in range(ybins):
            x = -x_size/2 + i*x_bin_width + x_bin_width/2.
            y = -y_size/2 + j*y_bin_width + y_bin_width/2.

            if np.sqrt((x-cx)**2 + (y-cy)**2) < sphere_r:
                sphere_bins.append(img2d[i, j])

    return np.average(sphere_bins)


def average_in_bckg(img2d : np.ndarray,
                    sphere_r : float, r : float,
                    phi_start : float, phi_step : float, n_phi : int,
                    x_size : float, y_size : float, xbins : int, ybins: int) -> float:
    """
    Calculates the average background density taking the average of n_phi spheres
    of radius sphere_r, centred at radius r and phi == phi_start + n*phi_step.
    """

    bckg_ave = 0
    for i in range(n_phi):
        bckg = average_in_sphere(img2d, sphere_r, r, phi_start + i*phi_step, x_size,
                                 y_size, xbins, ybins)
        bckg_ave += bckg

    bckg_ave /= n_phi

    return bckg_ave


def crc(img2d : np.ndarray, max_intensity : int,
        sig_sphere_r : float, r : float, phi : float,
        bckg_sphere_r : float, phi0 : float, phi_step : float, nphi : int,
        x_size : float, y_size : float, xbins : int, ybins : int) -> float:
    """
    Calculates the contrast_recovery_coefficent of an image.
    img2d: 2D np.array with the image.
    max_intensity: maximum value of original image pixels.
    sig_sphere_r: radius of the sphere of signal.
    r: radial position of both signal and background spheres.
    phi: angular position of signal sphere.
    bckg_sphere_r: radius of the sphere of background.
    phi0: angular position of first background sphere.
    phi_step: angular step of background spheres.
    nphi: number of backgrounds spheres used for average.
    x_size, y_size: size of the image in length unit.
    xbins, ybins: number of bins of the image.
    """

    signal = average_in_sphere(img2d, sig_sphere_r, r, phi, x_size, y_size, xbins, ybins)
    bckg   = average_in_bckg(img2d, bckg_sphere_r, r, phi0, phi_step, nphi,
                             x_size, y_size,  xbins, ybins)

    return signal / bckg / max_intensity
