import numpy as np

from scipy.special    import erf


def skewnormal_function(values, shape, location, scale, gain):
    '''
    Asymmetric gaussian function:
    Shape tells us the symmetry, the sideways shift of the
    center of the gaussine. Location is the center of the gaussine.
    Scale is the parameter relationated with the gaussian sigma.
    '''
    t   = (values - location) / scale

    pdf = np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)
    cdf = (1 + erf(shape * t / np.sqrt(2))) / 2

    return 2 * gain / scale * pdf * cdf


def linear_regression(x, slope, origin):
    '''Linear function'''

    return slope * x + origin


def linear_regression_inv_corr(x, slope, origin, shift):
    '''
    Inverted linear function with shift and period correction.
    '''
    period = 360

    possible = (x - origin) /slope - shift
    possible[possible < 0] = possible[possible < 0] + period

    return possible
