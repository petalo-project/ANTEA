import pandas           as pd
import numpy            as np
import os
import random

from scipy.stats      import skewnorm
from pytest           import raises

from antea.preproc.tdc_tfine_calibration import process_df_to_assign_tpulse_delays
from antea.preproc.tdc_tfine_calibration import compute_normalized_histogram
from antea.preproc.tdc_tfine_calibration import fit_gaussian
from antea.preproc.tdc_tfine_calibration import fit_semigaussian
from antea.preproc.tdc_tfine_calibration import one_distribution_fit
from antea.preproc.tdc_tfine_calibration import two_distributions_fit
from antea.preproc.tdc_tfine_calibration import select_fitting_distribution
from antea.preproc.tdc_tfine_calibration import fit_all_channel_phases
from antea.preproc.tdc_tfine_calibration import filter_anomalous_values_in_mode
from antea.preproc.tdc_tfine_calibration import TDC_linear_fit
from antea.preproc.tdc_tfine_calibration import TDC_double_linear_fit
from antea.preproc.tdc_tfine_calibration import fit_all_channel_modes


def test_process_df_to_assign_tpulse_delays(output_tmpdir):
    '''
    Check the correct dataframe result, the tofpet events array
    for each group of data and the number of wrong rows values and
    the channel_id they are.
    '''
    limits = pd.DataFrame({'start' : [0, 4, 7, 10, 14, 19],
                           'end'   : [4, 7, 10, 14, 19, 21],
                           'file1' : [0, 0, 0, 0, 1, 1],
                           'file2' : [0, 0, 0, 1, 1, 1]})

    configs = np.tile(np.arange(0, 30000, 10000), 2)

    fname_1 = os.path.join(output_tmpdir, 'test_tdc_calib_1.h5')
    fname_2 = os.path.join(output_tmpdir, 'test_tdc_calib_2.h5')
    files   = [fname_1, fname_2]

    df_1    = pd.DataFrame({'evt_number' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'run_control': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                            'tofpet_id'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'channel_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            'tfine'      : [317, 319, 315, 300, 305, 302,
                                            290, 293, 291, 315]})

    df_2    = pd.DataFrame({'evt_number' : [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                            'run_control': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                            'tofpet_id'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            'channel_id' : [1, 56, 1, 1, 1, 1, 1, 34, 1, 1],
                            'tfine'      : [317, 320, 319, 300, 303, 301, 299, 303,
                                            290, 291]})

    df_1.to_hdf(fname_1, key = 'data', format = 'table')
    df_2.to_hdf(fname_2, key = 'data', format = 'table')

    df, tofpet_evts, wrong_rows, wrong_ch_t = process_df_to_assign_tpulse_delays(files, limits, configs)

    df_2                 = df_2[df_2['channel_id'] == 1]
    df_expected          = pd.concat([df_1, df_2])
    df_expected['delay'] = [0, 0, 0, 10, 10, 10, 20, 20, 20, 0,
                            0, 0, 10, 10, 10, 10, 20, 20]

    tofpet_evts_expected = np.array([3, 3, 3, 3, 4, 2])
    wrong_rows_expected  =  2
    wrong_ch_t_expected  = [56, 34]


    np.testing.assert_array_equal(df, df_expected)
    np.testing.assert_array_equal(tofpet_evts, tofpet_evts_expected)
    np.testing.assert_array_equal(wrong_rows, wrong_rows_expected)
    np.testing.assert_array_equal(wrong_ch_t, wrong_ch_t_expected)


def test_compute_normalized_histogram():
    '''
    Check that the counts and edges obtained from a values
    array given are correct.
    '''
    bins       = 360
    hist_range = np.array([0, 360])
    values     = np.array([200, 201, 201, 202, 202, 202, 203, 203, 203, 203,
                           203, 204, 204, 204, 205, 205, 206])

    counts, xs      = compute_normalized_histogram(values, hist_range, bins)

    counts_expected = np.concatenate((np.tile(np.arange(1), 200), np.array([1, 2,
                      3, 5, 3, 2, 1])/17 , np.tile(np.arange(1), 153)), axis = 0)

    xs_expected     = np.arange(0.5, 360.5, 1)

    assert np.allclose(counts, counts_expected, atol = 0.001)
    assert np.allclose(xs, xs_expected, atol = 0.001)


def test_fit_gaussian():
    '''
    Check that the gaussian fit return the correct
    mean and sigma values.
    '''
    mu_expected    = 5
    sigma_expected = 1.5
    values         = np.random.normal(loc = mu_expected, scale = sigma_expected,
                                      size=10000).astype(int)

    mu, err_mu, sigma, err_sigma, chi2 = fit_gaussian(values)

    mu_tol    = 0.01 * mu_expected
    sigma_tol = 0.05 * sigma_expected

    assert np.allclose(mu, mu_expected, atol = mu_tol)
    assert np.allclose(sigma, sigma_expected, atol = sigma_tol)


def test_fit_semigaussian():
    '''
    Check that the skewnormal fit returns the correct
    mean, sigma and skew values.
    '''
    skew_expected  = -2
    loc_expected   = 250
    scale_expected = 2
    values         = []

    x            = np.linspace(loc_expected - 7,loc_expected + 7,14).astype(int)
    pdf_skewnorm = (skewnorm.pdf(x, skew_expected, loc_expected,
                                 scale_expected)*1000).astype(int)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm[i]))

        if len(rep) == 0:
            continue
        else:
            values.append(rep)

    values = np.concatenate(values)

    mu, err_mu, sigma, err_sigma, chi2, skew, loc, scale = fit_semigaussian(values)

    # Convert semigaussian expected location to gaussian:

    d         = skew_expected / np.sqrt(1 + skew_expected**2)
    comun_exp = np.sqrt(2 / np.pi) * d - (4 - np.pi)/4 * ((d*np.sqrt(2/np.pi))**3 /
                (1 - 2 * d**2 / np.pi)**(3/2) * np.sqrt(1 - 2 * d**2 /
                np.pi)) - np.sign(skew_expected) * np.exp(-2*np.pi /
                np.abs(skew_expected))/2

    mu_expected    = loc_expected + scale_expected * comun_exp

    loc_tol   = 0.01 * loc_expected
    scale_tol = 0.05 * scale_expected
    skew_tol  = 0.1 * np.abs(skew_expected)

    assert np.allclose(loc, loc_expected, atol = loc_tol)
    assert np.allclose(mu, mu_expected, atol = loc_tol)
    assert np.allclose(np.abs(scale), scale_expected, atol = scale_tol)
    assert np.allclose(skew, skew_expected, atol = skew_tol)


def test_one_distribution_fit():
    '''
    Check the data obtained after fitting to a single distribution.
    '''

    skew_expected  = -2
    loc_expected   = 250
    scale_expected = 2
    values         = []
    values_exp     = []

    x            = np.linspace(loc_expected - 7,loc_expected + 7,14).astype(int)
    pdf_skewnorm = (skewnorm.pdf(x, skew_expected, loc_expected,
                                 scale_expected)*1000).astype(int)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm[i]))

        if len(rep) == 0:
            continue
        else:
            values.append(rep)

    values = np.concatenate(values)

    fit_values, fit_errors, fit_chi = one_distribution_fit(values)

    # Convert semigaussian expected location to gaussian:
    d              = skew_expected / np.sqrt(1 + skew_expected**2)
    comun_exp      = np.sqrt(2/np.pi) * d - (4 - np.pi) / 4 * ((d*np.sqrt(2/np.pi))**3/
                     (1 - 2*d**2/np.pi)**(3/2) * np.sqrt(1 - 2*d**2/
                     np.pi)) - np.sign(skew_expected) * np.exp(-2*np.pi/
                     np.abs(skew_expected))/2
    mu_expected    = loc_expected + scale_expected * comun_exp


    loc_tol   = 0.01 * loc_expected
    scale_tol = 0.05 * scale_expected

    #Also, check the exception using experimental data:

    y_exp = np.array([1,   58,  364, 619, 1702, 917, 405, 155, 29,  4,   2 ])
    x_exp = np.array([187, 188, 189, 190, 191,  192, 193, 194, 195, 196, 197])

    for i in range(len(x_exp)):

        rep_exp = (np.tile(x_exp[i], y_exp[i]))

        if len(rep_exp) == 0:
            continue
        else:
            values_exp.append(rep_exp)

    values_exp = np.concatenate(values_exp)

    fit_values_exp, fit_errors_exp, fit_chi_exp = one_distribution_fit(values_exp)
    aprox_mode_exp = np.median(values_exp)

    assert np.allclose(fit_values[0], mu_expected,   atol = loc_tol)
    assert np.equal(fit_values[0], fit_values[2])
    assert np.allclose(fit_values_exp[0], aprox_mode_exp, atol = loc_tol)
    assert np.equal(fit_values_exp[0], fit_values_exp[2])


def test_two_distributions_fit():
    '''
    Check the data obtained after fitting to a double distribution.
    The data for checking the exceptions is experimental.
    '''

    ###Creating the double skewnorm distribution:
    skew_expected  = -2
    loc_expected_l = 180
    loc_expected_r = 300
    scale_expected = 2
    percentage     = 60

    values_l       = []
    values_r       = []
    values_eq      = []
    values_exp_l   = []
    values_exp_r   = []
    values_exp_eq  = []


    x_l = np.linspace(loc_expected_l - 7, loc_expected_l + 7, 14).astype(int)
    x_r = np.linspace(loc_expected_r - 7, loc_expected_r + 7, 14).astype(int)
    x   = np.concatenate((x_l, x_r), axis = 0)

    ## Left bigger distribution
    pdf_skewnorm_l_b = (skewnorm.pdf(x_l, skew_expected, loc_expected_l,
                                     scale_expected)*1000).astype(int)
    pdf_skewnorm_r_s = (skewnorm.pdf(x_r, skew_expected, loc_expected_r,
                                     scale_expected)*500).astype(int)
    pdf_skewnorm_l   = np.concatenate((pdf_skewnorm_l_b, pdf_skewnorm_r_s), axis = 0)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm_l[i]))

        if len(rep) == 0:
            continue
        else:
            values_l.append(rep)

    values_l = np.concatenate(values_l)

    ## Right bigger distribution
    pdf_skewnorm_l_s = (skewnorm.pdf(x_l, skew_expected, loc_expected_l,
                                     scale_expected)*500).astype(int)
    pdf_skewnorm_r_b = (skewnorm.pdf(x_r, skew_expected, loc_expected_r,
                                     scale_expected)*1000).astype(int)
    pdf_skewnorm_r   = np.concatenate((pdf_skewnorm_l_s, pdf_skewnorm_r_b), axis = 0)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm_r[i]))

        if len(rep) == 0:
            continue
        else:
            values_r.append(rep)

    values_r = np.concatenate(values_r)

    ## Equal distributions
    pdf_skewnorm_eq      = np.concatenate((pdf_skewnorm_l_b, pdf_skewnorm_r_b), axis = 0)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm_eq[i]))

        if len(rep) == 0:
            continue
        else:
            values_eq.append(rep)

    values_eq = np.concatenate(values_eq)

    ### Experimental data for checking the exception:

    y_exp = np.array([1,   58,  364, 619, 1702, 917, 405, 155, 29,  4,   2 ])
    x_exp = np.array([187, 188, 189, 190, 191,  192, 193, 194, 195, 196, 197])

    ## Left bigger distribution
    pdf_skewnorm_l_exp = np.concatenate((y_exp, pdf_skewnorm_r_b), axis = 0)
    x_exp_l_all        = np.concatenate((x_exp, x_r), axis = 0)

    for i in range(len(x_exp_l_all)):

        rep_exp = (np.tile(x_exp_l_all[i], pdf_skewnorm_l_exp[i]))

        if len(rep_exp) == 0:
            continue
        else:
            values_exp_l.append(rep_exp)

    values_exp_l = np.concatenate(values_exp_l)

    ## Right bigger distribution
    pdf_skewnorm_r_exp = np.concatenate((pdf_skewnorm_l_b, y_exp), axis = 0)
    x_exp_r_all        = np.concatenate((x_l, x_exp + 100), axis = 0)

    for i in range(len(x_exp_r_all)):

        rep_exp = (np.tile(x_exp_r_all[i], pdf_skewnorm_r_exp[i]))

        if len(rep_exp) == 0:
            continue
        else:
            values_exp_r.append(rep_exp)

    values_exp_r = np.concatenate(values_exp_r)

    ## Equal distributions
    pdf_skewnorm_eq_exp = np.concatenate((y_exp, y_exp), axis = 0)
    x_exp_eq_all        = np.concatenate((x_exp, x_exp + 100), axis = 0)

    for i in range(len(x_exp_eq_all)):

        rep_exp = (np.tile(x_exp_eq_all[i], pdf_skewnorm_eq_exp[i]))

        if len(rep_exp) == 0:
            continue
        else:
            values_exp_eq.append(rep_exp)

    values_exp_eq = np.concatenate(values_exp_eq)


    fit_vals_l,      _, _ = two_distributions_fit(values_l, percentage)
    fit_vals_r,      _, _ = two_distributions_fit(values_r, percentage)
    fit_vals_eq,     _, _ = two_distributions_fit(values_eq, percentage)
    fit_vals_exp_l,  _, _ = two_distributions_fit(values_exp_l, percentage)
    fit_vals_exp_r,  _, _ = two_distributions_fit(values_exp_r, percentage)
    fit_vals_exp_eq, _, _ = two_distributions_fit(values_exp_eq, percentage)

    # Convert semigaussian expected location to gaussian:
    d     = skew_expected / np.sqrt(1 + skew_expected**2)
    comun = np.sqrt(2/np.pi)*d - (4 - np.pi)/4*((d * np.sqrt(2/np.pi))**3/
            (1 - 2 * d**2/np.pi)**(3/2) * np.sqrt(1 - 2*d**2/
            np.pi)) - np.sign(skew_expected)*np.exp(-2*np.pi/np.abs(skew_expected))/2

    mu_expected_l = loc_expected_l + scale_expected * comun
    mu_expected_r = loc_expected_r + scale_expected * comun

    aprox_mode_exp_l = np.median(x_exp)
    aprox_mode_exp_r = np.median(x_exp + 100)

    loc_tol_l = 0.01 * loc_expected_l
    loc_tol_r = 0.01 * loc_expected_r
    scale_tol = 0.05 * scale_expected

    assert np.allclose(fit_vals_l[0],      mu_expected_l,     atol = loc_tol_l)
    assert np.allclose(fit_vals_l[0],      fit_vals_l[2],     atol = loc_tol_l)

    assert np.allclose(fit_vals_r[0],      mu_expected_r,     atol = loc_tol_r)
    assert np.allclose(fit_vals_r[0],      fit_vals_r[2],     atol = loc_tol_r)

    assert np.allclose(fit_vals_eq[0],     mu_expected_l,     atol = loc_tol_l)
    assert np.allclose(fit_vals_eq[2],     mu_expected_r,     atol = loc_tol_r)

    assert np.allclose(fit_vals_exp_l[0],  aprox_mode_exp_l,  atol = loc_tol_l)
    assert np.allclose(fit_vals_exp_l[0],  fit_vals_exp_l[2], atol = loc_tol_l)

    assert np.allclose(fit_vals_exp_r[0],  aprox_mode_exp_r,  atol = loc_tol_r)
    assert np.allclose(fit_vals_exp_r[0],  fit_vals_exp_r[2], atol = loc_tol_r)

    assert np.allclose(fit_vals_exp_eq[0], aprox_mode_exp_l,  atol = loc_tol_l)
    assert np.allclose(fit_vals_exp_eq[2], aprox_mode_exp_r,  atol = loc_tol_r)


def test_select_fitting_distribution():
    '''
    Check we obtain the correct results when fitting
    one semigaussian distribution or two semigaussian distribution.
    It also checks the error exception.
    '''
    ###Creating data for one skewnorm distribution:

    skew_expected  = -2
    loc_expected   = 250
    scale_expected = 2
    values         = []

    x            = np.linspace(loc_expected - 7,loc_expected + 7,14).astype(int)
    pdf_skewnorm = (skewnorm.pdf(x, skew_expected, loc_expected,
                                 scale_expected)*1000).astype(int)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm[i]))

        if len(rep) == 0:
            continue
        else:
            values.append(rep)

    values = np.concatenate(values)

    ###Creating the double skewnorm distribution:

    loc_expected_l = 180
    loc_expected_r = 300
    percentage     = 60
    values_double  = []

    x_l      = np.linspace(loc_expected_l - 7, loc_expected_l + 7, 14).astype(int)
    x_r      = np.linspace(loc_expected_r - 7, loc_expected_r + 7, 14).astype(int)
    x_double = np.concatenate((x_l, x_r), axis = 0)

    ## Left bigger distribution
    pdf_skewnorm_l_b = (skewnorm.pdf(x_l, skew_expected, loc_expected_l,
                                 scale_expected)*1000).astype(int)
    pdf_skewnorm_r_s = (skewnorm.pdf(x_r, skew_expected, loc_expected_r,
                                 scale_expected)*500).astype(int)
    pdf_skewnorm_l   = np.concatenate((pdf_skewnorm_l_b, pdf_skewnorm_r_s), axis = 0)

    for i in range(len(x_double)):

        rep_double = (np.tile(x_double[i], pdf_skewnorm_l[i]))

        if len(rep_double) == 0:
            continue
        else:
            values_double.append(rep_double)

    values_double = np.concatenate(values_double)

    ###Creating no skewnorm distribution for exception
    y_unif_1      = np.repeat(np.linspace(100, 200, 101), 100)
    y_unif_2      = np.repeat(np.linspace(300, 400, 101), 300)
    y_unif_double = np.concatenate((y_unif_1, y_unif_2), axis = 0)


    fit_res        = select_fitting_distribution(values, percentage)
    fit_res_double = select_fitting_distribution(values_double, percentage)

    # Convert semigaussian expected location to gaussian:
    d         = skew_expected / np.sqrt(1 + skew_expected**2)
    comun     = np.sqrt(2/np.pi)*d - (4 - np.pi)/4*((d * np.sqrt(2/np.pi))**3/
                        (1 - 2*d**2/np.pi)**(3/2)*np.sqrt(1 - 2*d**2/np.pi)) - np.sign(
                    skew_expected)*np.exp(-2*np.pi/np.abs(skew_expected))/2

    mu_expected_l = loc_expected_l + scale_expected * comun
    mu_expected   = loc_expected   + scale_expected * comun

    loc_tol_l = 0.01 * loc_expected_l
    loc_tol   = 0.01 * loc_expected

    assert np.allclose(fit_res_double[0][0], mu_expected_l,        atol = loc_tol_l)
    assert np.allclose(fit_res_double[0][0], fit_res_double[0][2], atol = loc_tol_l)

    assert np.allclose(fit_res[0][0],        mu_expected,          atol = loc_tol)
    assert np.allclose(fit_res[0][0],        fit_res[0][2],        atol = loc_tol)

    with raises(RuntimeError):
        select_fitting_distribution(y_unif_1, percentage)

    with raises(RuntimeError):
        select_fitting_distribution(y_unif_double, percentage)

def test_fit_all_channel_phases(output_tmpdir):
    '''
    Check the values obtained in the DataFrame are correct.
    '''
    ## Create data for file:

    skew_expected  = -2
    loc_expected_l = 180
    loc_expected_r = 300
    scale_expected = 2

    values_l = []
    values   = []

    # Double distribution:
    x_l = np.linspace(loc_expected_l - 7, loc_expected_l + 7, 14).astype(int)
    x_r = np.linspace(loc_expected_r - 7, loc_expected_r + 7, 14).astype(int)
    x   = np.concatenate((x_l, x_r), axis = 0)

    pdf_skewnorm_l_b = (skewnorm.pdf(x_l, skew_expected, loc_expected_l,
                                 scale_expected)*1000).astype(int)
    pdf_skewnorm_r_s = (skewnorm.pdf(x_r, skew_expected, loc_expected_r,
                                 scale_expected)*500).astype(int)
    pdf_skewnorm_l   = np.concatenate((pdf_skewnorm_l_b, pdf_skewnorm_r_s), axis = 0)

    for i in range(len(x)):

        rep = (np.tile(x[i], pdf_skewnorm_l[i]))

        if len(rep) == 0:
            continue
        else:
            values_l.append(rep)

    values_l = np.concatenate(values_l)

    # One distribution:
    for i in range(len(x_r)):

        rep_r = (np.tile(x_r[i], pdf_skewnorm_r_s[i]))

        if len(rep_r) == 0:
            continue
        else:
            values.append(rep_r)

    values = np.concatenate(values)

    ## Create file:
    filein = os.path.join(output_tmpdir, 'tdc_phases.h5')

    df_0   = pd.DataFrame({'channel_id'  : 0,
                           'tac_id'      : 0,
                           'delay'       : 0.0,
                           'tfine'       : values_l})

    df_01  = pd.DataFrame({'channel_id' : 0,
                           'tac_id'     : 1,
                           'delay'      : 6.0,
                           'tfine'      : values})

    df_0   = df_0.append(df_01)
    df_1   = df_0.copy()
    df_1['channel_id'] = 1

    df_0.to_hdf(filein, key = 'ch0')
    df_1.to_hdf(filein, key = 'ch1')

    channels   = np.arange(2)
    percentage = 60

    df_tfine = fit_all_channel_phases(filein, channels, percentage)

    df_tfine_expected = pd.DataFrame({'channel_id' : [0, 0 ,1, 1],
                                      'tac_id'     : [0, 1, 0, 1],
                                      'phase'      : [0.0, 6.0, 0.0, 6.0],
                                      'mode_l'     : [178.93, 298.93, 178.93, 298.93],
                                      'mode_r'     : [178.93, 298.93, 178.93, 298.93]})

    loc_tol_l = 0.01 * loc_expected_l
    loc_tol_r = 0.01 * loc_expected_r

    assert np.allclose(df_tfine['channel_id'], df_tfine_expected['channel_id'])
    assert np.allclose(df_tfine['tac_id']    , df_tfine_expected['tac_id'])
    assert np.allclose(df_tfine['phase']     , df_tfine_expected['phase'])
    assert np.allclose(df_tfine['mode_l']    , df_tfine_expected['mode_l'], loc_tol_l)
    assert np.allclose(df_tfine['mode_r']    , df_tfine_expected['mode_r'], loc_tol_r)

def test_filter_anomalous_values_in_mode():
    '''
    Check if the correction done in mode values obtained
    from the skewnorm fit is correct.
    '''

    df = pd.DataFrame({'channel_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'tac_id'     : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'phase'      : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                       'mode_l'     : [350.7, 237.5, 241.2, 342.0, 339.7, 337.3,
                                       334.6, 223.9, 325.2, 326.2],
                       'err_mode_l' : [0.317, 0.519, 0.515, 0.300, 0.305, 3020,
                                       0.290, 0.793, 0.691, 0.315],

                       'mode_r'     : [350.7, 347.5, 345.2, 342.0, 339.7, 337.3,
                                       334.6, 331.9, 328.2, 326.2],
                       'err_mode_r' : [0.317, 0.219, 0.215, 0.300, 0.305, 3020,
                                       0.290, 0.593, 0.691, 0.315]})

    df_expected =  pd.DataFrame({'phase'      : [0, 10, 20, 30, 40, 60, 70, 80, 90],
                                 'mode_l'     : [350.7, 237.5, 241.2, 342.0, 339.7,
                                                 334.6, 223.9, 325.2, 326.2],
                                 'err_mode_l' : [0.317, 0.519, 0.515, 0.300, 0.305,
                                                 0.290, 0.793, 0.691, 0.315],
                                 'mode_r'     : [350.7, 347.5, 345.2, 342.0, 339.7,
                                                 334.6, 331.9, 328.2, 326.2],
                                 'err_mode_r' : [0.317, 0.219, 0.215, 0.300, 0.305,
                                                 0.290, 0.593, 0.691, 0.315],
                                 'mode'       : [350.7, 347.5, 345.2, 342.0, 339.7,
                                                 334.6, 331.9, 325.2, 326.2],
                                 'err_mode'   : [0.317, 0.219, 0.215, 0.300, 0.305,
                                                 0.290, 0.593, 0.691, 0.315],
                                 'diff'       : [0, -113.2, 3.7, 100.8, -2.3,-5.1, -110.7,
                                                 101.3, 1.0]})

    df = filter_anomalous_values_in_mode(df)
    assert np.allclose(df, df_expected)

def test_TDC_linear_fit():
    '''
    Check that the linear fit returns the correct coefficients.
    '''

    slope  = -0.42
    origin = 238
    mu     = 0
    sigma  = 0.5
    y_gaus = []
    x      = np.random.randint(0, 175, 25)

    # Add a gaussian error to all values
    for i in range(25):
        gaus = random.gauss(mu, sigma)
        y_gaus.append(gaus)

    y     = slope*x + origin + y_gaus
    y_val = slope*30 + origin

    df = pd.DataFrame()
    df['phase'] = x
    df['mode']  = y

    coeff, coeff_err, chisq_r, func = TDC_linear_fit(df)

    tol_origin = 0.01 * origin
    tol_val    = 0.01 * y_val

    assert np.allclose(slope,  coeff[0], atol = 0.01)
    assert np.allclose(origin, coeff[1], atol = tol_origin)
    assert np.allclose(y_val,  func(30), atol = tol_val)

def test_TDC_double_linear_fit():
    '''
    Check the correct results of the linear coefficients,
    the shift point and the data frames obtained from
    the two linear fits.
    '''

    ###Create both linear distributions:

    # Parameters lineal low:
    slope_low   = -0.42
    origin_low  = 238
    x_low       = np.concatenate([[0], np.random.randint(0, 175, 25), [175]], axis = 0)

    # Parameters lineal high:
    slope_high  = -0.42
    origin_high = 380
    x_high      = np.concatenate([[180], np.random.randint(180, 360, 25), [360]], axis = 0)

    # Add a gaussian error to all values
    mu     = 0
    sigma  = 0.5
    y_gaus = []

    for i in range(27):
        gaus = random.gauss(mu, sigma)
        y_gaus.append(gaus)

    y_low  = slope_low  * x_low  + origin_low  + y_gaus
    y_high = slope_high * x_high + origin_high + y_gaus

    df_low = pd.DataFrame({'channel_id' : 0,
                           'tac_id'     : 0,
                           'phase'      : x_low,
                           'mode'       : y_low})

    df_high = pd.DataFrame({'channel_id' : 0,
                            'tac_id'     : 0,
                            'phase'      : x_high,
                            'mode'       : y_high})

    df = pd.concat([df_low, df_high]).reset_index(drop = True)

    channel = 0
    tac     = 0
    coeff, coeff_err, shifts_vals, chisq_r, dfs = TDC_double_linear_fit(df, channel, tac)

    shift_ph_expected   = 177.5
    shift_mode_expected = 234.45

    tol_orig_low   = 0.01 * origin_low
    tol_orig_high  = 0.01 * origin_high
    tol_shift_ph   = 0.01 * shift_ph_expected
    tol_shift_mode = 0.01 * shift_mode_expected

    assert np.allclose(slope_low,   coeff[0][0], atol = 0.01)
    assert np.allclose(origin_low,  coeff[0][1], atol = tol_orig_low)
    assert np.allclose(slope_high,  coeff[1][0], atol = 0.01)
    assert np.allclose(origin_high, coeff[1][1], atol = tol_orig_high)

    assert np.allclose(shift_ph_expected, shifts_vals[0], atol = tol_shift_ph)
    assert np.allclose(shift_mode_expected, shifts_vals[1], atol = tol_shift_mode)

    assert np.allclose(df_low, dfs[0])
    assert np.allclose(df_high, dfs[1])

def test_fit_all_channel_modes(output_tmpdir):
    '''
    Check the values obtained in the DataFrame after the double linear
    fit are correct.
    '''

    df = pd.DataFrame({'channel_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'tac_id'     : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'phase'      : [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                       'mode_l'     : [40, 30, 20, 10, 0, 140, 130, 120, 110, 100],
                       'err_mode_l' : [0.317, 0.219, 0.215, 0.300, 0.305, 0.020,
                                       0.290, 0.593, 0.691, 0.315],
                       'mode_r'     : [40, 30, 20, 10, 0, 140, 130, 120, 110, 100],
                       'err_mode_r' : [0.317, 0.219, 0.215, 0.300, 0.305, 0.020,
                                       0.290, 0.593, 0.691, 0.315]})

    df1 = df.copy()
    df1['tac_id']     = 1
    df2 = df.copy()
    df2['channel_id'] = 1
    df3 = df.copy()
    df3['tac_id']     = 1
    df3['channel_id'] = 1

    dfs = pd.concat([df, df1, df2, df3]).reset_index(drop = True)

    ## Create file:
    filein = os.path.join(output_tmpdir, 'tdc_modes.h5')
    dfs.to_hdf(filein, key = 'tfine')

    df_tfine_linear = fit_all_channel_modes(filein)

    df_linear_expected = pd.DataFrame({'channel_id'  : [0, 0, 1, 1],
                                       'tac_id'      : [0, 1, 0, 1],
                                       'slope_low'   : -1,
                                       'origin_low'  : 40,
                                       'slope_high'  : -1,
                                       'origin_high' : 190,
                                       'shift_phase' : 45,
                                       'shift_mode'  : 70})

    assert np.allclose(df_linear_expected.channel_id,  df_tfine_linear.channel_id)
    assert np.allclose(df_linear_expected.tac_id,      df_tfine_linear.tac_id)
    assert np.allclose(df_linear_expected.slope_low,   df_tfine_linear.slope_low)
    assert np.allclose(df_linear_expected.origin_low,  df_tfine_linear.origin_low)
    assert np.allclose(df_linear_expected.slope_high,  df_tfine_linear.slope_high)
    assert np.allclose(df_linear_expected.origin_high, df_tfine_linear.origin_high)
    assert np.allclose(df_linear_expected.shift_phase, df_tfine_linear.shift_phase)
    assert np.allclose(df_linear_expected.shift_mode,  df_tfine_linear.shift_mode)
