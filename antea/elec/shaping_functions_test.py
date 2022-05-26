import os
import numpy                 as np
import hypothesis.strategies as st

from scipy.signal  import fftconvolve
from numpy.testing import assert_almost_equal

from hypothesis     import given
from pytest         import mark
from antea.io.mc_io import load_mcTOFsns_response
from antea.elec     import shaping_functions   as shf

from invisible_cities.core import system_of_units as units


tau_sipm = [100, 15000]
l        = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l)
def test_normalize_sipm_shaping(l):
    """
    This test checks that the function normalize_sipm_shaping returns an array
    with the distribution value for each time.
    """
    l = np.array(l)
    exp_dist, _ = shf.normalize_sipm_shaping(np.unique(l), tau_sipm)

    assert len(exp_dist) == len(np.unique(l))
    assert (exp_dist >= 0.).all()
    assert np.isclose(np.sum(exp_dist), 1)


@mark.parametrize('time, time_dist',
                  ((  0, 0),
                   (100, 0.65120889),
                   (np.array([1,2,3]), np.array([0.01029012, 0.02047717, 0.03056217]))))
def test_sipm_shaping(time, time_dist):
    """
    sipm_shaping is an analitic function, so this test takes some values
    and checks that the function returns the correct value for each one.
    """
    result = shf.sipm_shaping(time, tau_sipm)
    assert np.all(result) == np.all(time_dist)


def test_elec_shaping():
    """
    Check that the function returns the correct result
    for specific values of the variable.
    """
    result = shf.elec_shaping(np.array([0, 252.69]))
    assert np.all(result) == np.all(np.array([5.065e-6, 5.065e-6/np.e]))



s = st.lists(st.integers(min_value=1, max_value=10000), min_size=2, max_size=1000)

@given(l, s)
def test_convolve_sipm_shaping(l, s):
    """
    Check that the function convolve_signal_with_shaping returns an array with the adequate length,
    and, in case the array is not empty,
    checks that the convoluted signal is normalizated to the initial signal.
    """
    spe_response, _ = shf.normalize_sipm_shaping(np.unique(np.array(l)), tau_sipm)
    conv_res           = shf.convolve_signal_with_shaping(np.array(s), spe_response)
    assert len(conv_res) == len(spe_response) + len(s) - 1
    if np.count_nonzero(spe_response):
        assert np.isclose(np.sum(s), np.sum(conv_res))


@mark.parametrize('filename',
                  (('ring_test.h5'),
                   ('full_body_1ev.h5')))
def test_sipm_shaping_convolution(ANTEADATADIR, filename):
    """
    Check that the function sipm_shaping_convolution
    returns a table with the adequate dimensions
    and that the table always contains non-zero values
    if there are times in the window.
    """
    PATH_IN        = os.path.join(ANTEADATADIR, filename)
    tof_response   = load_mcTOFsns_response(PATH_IN)
    events         = tof_response.event_id.unique()
    time_window    = 5000
    tof_bin_size   = 5 * units.ps
    time           = np.arange(0, time_window)
    spe_resp, _    = shf.normalize_sipm_shaping(time, tau_sipm)

    for evt in events:
        evt_tof = tof_response[tof_response.event_id == evt]
        times   = evt_tof.time_bin.values * tof_bin_size / units.ps
        evt_tof.insert(4, 'time', times.astype(int))
        tof_sns = evt_tof.sensor_id.unique()
        for s_id in tof_sns:
            tdc_conv = shf.sipm_shaping_convolution(evt_tof, spe_resp, s_id, time_window)
            assert len(tdc_conv) == time_window + len(spe_resp) - 1
            if len(evt_tof[(evt_tof.sensor_id == s_id) &
                         (evt_tof.time >= time_window)]) == 0:
                assert np.count_nonzero(tdc_conv) > 0


e     = st.integers(min_value=0,     max_value= 1000)
s_id  = st.integers(min_value=-3500, max_value=-1000)
l2    = st.lists(st.floats(min_value=0, max_value=1000), min_size=2, max_size=100)

@given(e, s_id, l2)
def test_build_convoluted_df(e, s_id, l2):
    """
    Look whether the build_convoluted_df function returns a dataframe
    with the same number of rows as the input numpy array and four columns.
    Three of this columns must contain integers.
    """
    l2    = np.array(l2)
    wf_df = shf.build_convoluted_df(e, s_id, l2)
    assert len(wf_df) == np.count_nonzero(l2)
    assert len(wf_df.keys()) == 4
    if np.count_nonzero(l2) == 0:
        assert wf_df.empty
    else:
        assert wf_df.event_id .dtype == 'int32'
        assert wf_df.sensor_id.dtype == 'int32'
        assert wf_df.time .dtype     == 'int32'


@given(l, s)
def test_convolve_fftconvolve_equivalent(l, s):
    """
    Check that the fftconvolve function returns
    the same array as convolve_signal_with_shaping.
    """
    spe_response, _ = shf.normalize_sipm_shaping(np.unique(np.array(l)), tau_sipm)
    conv_res        = shf.convolve_signal_with_shaping(np.array(s), spe_response)
    fftconv_res     = fftconvolve(np.array(s), spe_response)
    assert_almost_equal(conv_res, fftconv_res)
