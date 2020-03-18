import numpy as np

from typing import Tuple

"""
Functions that provide figure of merits for image reconstruction.
The image is assumed to be a phantom with spheres of different radii on
a background.
"""


def mean_error_in_sphere2d(img2d : np.ndarray,
                           sphere_r : float, r : float, phi : float,
                           x_size : float, y_size : float,
                           xbins : int, ybins : int) -> Tuple[float, float]:
    """
    Calculates the mean density in a circle of radius sphere_r,
    centred in (r, phi) and the error on the mean.
    x_size, y_size: size of the image (in length unit).
    xbins, ybins: number of bins  in x and y.

    :param img2d: 2D np.array with the image.
    :param sphere_r: radius of the sphere.
    :param r: radial position of the sphere.
    :param phi: angular position of sphere.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    """

    x_bin_width = x_size / xbins
    y_bin_width = y_size / ybins
    cx = r*np.cos(phi)
    cy = r*np.sin(phi)

    i = np.array(range(xbins))
    j = np.array(range(ybins))

    x = -x_size/2 + i*x_bin_width + x_bin_width/2.
    y = -y_size/2 + j*y_bin_width + y_bin_width/2.

    comb_xy = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    comb_ij = np.array(np.meshgrid(i, j)).T.reshape(-1, 2)

    in_sphere = np.sqrt((comb_xy[:, 0]-cx)**2 + (comb_xy[:, 1]-cy)**2) < sphere_r
    img_bins  = comb_ij[in_sphere]

    circle = img2d[img_bins[:, 0], img_bins[:, 1]]

    return np.average(circle), np.std(circle)/len(circle)


def mean_error_in_sphere3d(img3d : np.ndarray,
                           sphere_r : float, r : float, phi : float,
                           x_size : float, y_size : float, z_size : float,
                           xbins : int, ybins : int, zbins : int) -> Tuple[float, float]:
    """
    Calculates the mean density in a sphere of radius sphere_r,
    centred in (r, phi) and the error on the mean.
    x_size, y_size, z_size: size of the image (in length unit).
    xbins, ybins, zbins: number of bins in x, y and z.

    :param img3d: 3D np.array with the image.
    :param sphere_r: radius of the sphere.
    :param r: radial position of the sphere.
    :param phi: angular position of the sphere.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param z_size: z size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    :param zbins: number of bins of the image in the z axis.
    """

    x_bin_width = x_size / xbins
    y_bin_width = y_size / ybins
    z_bin_width = z_size / zbins
    cx = r*np.cos(phi)
    cy = r*np.sin(phi)
    cz = 0.

    i = np.array(range(xbins))
    j = np.array(range(ybins))
    k = np.array(range(zbins))

    x = -x_size/2 + i*x_bin_width + x_bin_width/2.
    y = -y_size/2 + j*y_bin_width + y_bin_width/2.
    z = -z_size/2 + k*z_bin_width + z_bin_width/2.

    comb_xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    comb_ijk = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3)

    in_sphere = np.sqrt((comb_xyz[:, 0] - cx)**2 + (comb_xyz[:, 1] - cy)**2 +
                        (comb_xyz[:, 2] - cz)**2) < sphere_r
    img_bins  = comb_ijk[in_sphere]

    sphere = img3d[img_bins[:, 0], img_bins[:, 1], img_bins[:, 2]]

    return np.average(sphere), np.std(sphere)/len(sphere)


def average_mean_error_in_bckg2d(img2d : np.ndarray,
                                 sphere_r : float, r : float,
                                 phi_start : float, phi_step : float,
                                 n_phi : int, x_size : float, y_size : float,
                                 xbins : int, ybins: int)-> Tuple[float, float,
                                                                  float, float]:
    """
    Calculates the average mean and error on the mean of background
    taking the average of the means (and their errors) of n_phi circles
    of radius sphere_r, centred at radius r and phi == phi_start + n*phi_step.

    :param img2d: 2D np.array with the image.
    :param sphere_r: radius of the background spheres.
    :param r: radial position where background spheres are placed.
    :param phi_start: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    """

    mean_bckgs, err_bckgs = [], []
    for i in range(n_phi):
        mean_bckg, err_bckg = mean_error_in_sphere2d(img2d, sphere_r, r,
                                                     phi_start + i*phi_step, x_size,
                                                     y_size, xbins, ybins)

        mean_bckgs.append(mean_bckg)
        err_bckgs.append(err_bckg)

    return np.average(mean_bckgs), np.std(mean_bckgs)/n_phi, np.average(err_bckgs), np.std(err_bckgs)/n_phi


def average_mean_error_in_bckg3d(img3d : np.ndarray,
                                 sphere_r : float, r : float,
                                 phi_start : float, phi_step : float,
                                 n_phi : int, x_size : float, y_size : float,
                                 z_size : float, xbins : int, ybins: int,
                                 zbins : int) -> Tuple[float, float, float, float]:
    """
    Calculates the average mean and error on the mean of background
    taking the average of the means (and their errors) of n_phi spheres
    of radius sphere_r, centred at radius r and phi == phi_start + n*phi_step.

    :param img3d: 3D np.array with the image.
    :param sphere_r: radius of the background spheres.
    :param r: radial position where background spheres are placed.
    :param phi_start: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param z_size: z size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    :param zbins: number of bins of the image in the z axis.
    """

    mean_bckgs, err_bckgs = [], []
    for i in range(n_phi):
        mean_bckg, err_bckg = mean_error_in_sphere3d(img3d, sphere_r, r,
                                                     phi_start + i*phi_step,
                                                     x_size, y_size, z_size,
                                                     xbins, ybins, zbins)

        mean_bckgs.append(mean_bckg)
        err_bckgs.append(err_bckg)

    return np.average(mean_bckgs), np.std(mean_bckgs)/n_phi, np.average(err_bckgs), np.std(err_bckgs)/n_phi


def crc_hot2d(img2d : np.ndarray, true_signal : float, true_bckg : float,
              sig_sphere_r : float, r : float, phi : float,
              bckg_sphere_r : float, phi0 : float, phi_step : float, nphi : int,
              x_size : float, y_size : float, xbins : int, ybins : int) -> float:
    """
    Calculates the contrast_recovery_coefficent of hot circle.

    :param img2d: 2D np.array with the image.
    :param true_signal: maximum value of original image pixels.
    :param true_bckg: value of background pixels.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    """

    signal, _ = mean_error_in_sphere2d(img2d, sig_sphere_r, r, phi,
                                       x_size, y_size, xbins, ybins)
    bckg, _, _, _   = average_mean_error_in_bckg2d(img2d, bckg_sphere_r, r,
                                                   phi0, phi_step, nphi,
                                                   x_size, y_size,  xbins, ybins)

    alpha = true_signal / true_bckg

    return (signal/bckg - 1)/(alpha - 1)


def crc_hot3d(img3d : np.ndarray, true_signal : float, true_bckg : float,
              sig_sphere_r : float, r : float, phi : float,
              bckg_sphere_r : float, phi0 : float, phi_step : float, nphi : int,
              x_size : float, y_size : float, z_size : float,
              xbins : int, ybins : int, zbins : int):
    """
    Calculates the contrast_recovery_coefficent of hot sphere.

    :param img3d: 3D np.array with the image.
    :param true_signal: maximum value of original image pixels.
    :param true_bckg: value of background pixels.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param z_size: z size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    :param zbins: number of bins of the image in the z axis.
    """

    signal, _ = mean_error_in_sphere3d(img3d, sig_sphere_r, r, phi,
                                       x_size, y_size, z_size,
                                       xbins, ybins, zbins)
    bckg, _, _, _   = average_mean_error_in_bckg3d(img3d, bckg_sphere_r, r, phi0,
                                                   phi_step, nphi, x_size, y_size,
                                                   z_size, xbins, ybins, zbins)

    alpha = true_signal / true_bckg

    return (signal/bckg - 1)/(alpha - 1)



def crc_cold2d(img2d : np.ndarray, sig_sphere_r : float, r : float, phi : float,
               bckg_sphere_r : float, phi0 : float, phi_step : float,
               nphi : int, x_size : float, y_size : float, xbins : int,
               ybins : int) -> float:
    """
    Calculates the contrast_recovery_coefficent of a cold circle.

    :param img2d: 2D np.array with the image.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    """

    signal, _ = mean_error_in_sphere2d(img2d, sig_sphere_r, r, phi,
                                       x_size, y_size, xbins, ybins)
    bckg, _, _, _   = average_mean_error_in_bckg2d(img2d, bckg_sphere_r, r, phi0,
                                                   phi_step, nphi, x_size, y_size,
                                                   xbins, ybins)

    return 1 - signal / bckg


def crc_cold3d(img3d : np.ndarray, sig_sphere_r : float, r : float, phi : float,
               bckg_sphere_r : float, phi0 : float, phi_step : float,
               nphi : int, x_size : float, y_size : float, z_size : float,
               xbins : int, ybins : int, zbins : int) -> float:
    """
    Calculates the contrast_recovery_coefficent of a cold sphere.

    :param img3d: 3D np.array with the image.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param z_size: z size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    :param zbins: number of bins of the image in the z axis.
    """

    signal, _ = mean_error_in_sphere3d(img3d, sig_sphere_r, r, phi,
                                       x_size, y_size, z_size,
                                       xbins, ybins, zbins)
    bckg, _, _, _   = average_mean_error_in_bckg3d(img3d, bckg_sphere_r, r, phi0,
                                                   phi_step, nphi, x_size, y_size,
                                                   z_size, xbins, ybins, zbins)

    return 1 - signal / bckg


def snr2d(img2d : np.ndarray,
          sig_sphere_r : float, r : float, phi : float,
          bckg_sphere_r : float, phi0 : float, phi_step : float, nphi : int,
          x_size : float, y_size : float,
          xbins : int, ybins : int) -> float:
    """
    Calculates the signal to noise ratio of a 2D image.

    :param img2d: 2D np.array with the image.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    """

    signal, _ = mean_error_in_sphere2d(img2d, sig_sphere_r, r, phi,
                                       x_size, y_size, xbins, ybins)
    bckg, err_bckg, _, _ = average_mean_error_in_bckg2d(img2d, bckg_sphere_r, r,
                                                        phi0, phi_step, nphi,
                                                        x_size, y_size,  xbins,
                                                        ybins)

    return (signal - bckg) / err_bckg


def snr3d(img3d : np.ndarray,
          sig_sphere_r : float, r : float, phi : float,
          bckg_sphere_r : float, phi0 : float, phi_step : float, nphi : int,
          x_size : float, y_size : float, z_size : float,
          xbins : int, ybins : int, zbins : int) -> float:
    """
    Calculates the signal to noise ratio of a 3D image.

    :param img3d: 3D np.array with the image.
    :param sig_sphere_r: radius of the sphere of signal.
    :param r: radial position of both signal and background spheres.
    :param phi: angular position of signal sphere.
    :param bckg_sphere_r: radius of the sphere of background.
    :param phi0: angular position of first background sphere.
    :param phi_step: angular step of background spheres.
    :param nphi: number of backgrounds spheres used for average.
    :param x_size: x size of the image in length unit.
    :param y_size: y size of the image in length unit.
    :param z_size: z size of the image in length unit.
    :param xbins: number of bins of the image in the x axis.
    :param ybins: number of bins of the image in the y axis.
    :param zbins: number of bins of the image in the z axis.
    """

    signal, _ = mean_error_in_sphere3d(img3d, sig_sphere_r, r, phi,
                                       x_size, y_size, z_size,
                                       xbins, ybins, zbins)
    bckg, err_bckg, _, _ = average_mean_error_in_bckg3d(img3d, bckg_sphere_r, r,
                                                        phi0, phi_step, nphi,
                                                        x_size, y_size, z_size,
                                                        xbins, ybins, zbins)

    return (signal - bckg) / err_bckg
