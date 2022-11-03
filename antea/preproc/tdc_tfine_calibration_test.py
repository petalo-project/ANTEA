import pandas           as pd
import numpy            as np
import os
import random

from scipy.stats      import skewnorm

from antea.preproc.tdc_tfine_calibration import process_df_to_assign_tpulse_delays
from antea.preproc.tdc_tfine_calibration import compute_normalized_histogram
from antea.preproc.tdc_tfine_calibration import fit_gaussian
from antea.preproc.tdc_tfine_calibration import fit_semigaussian
from antea.preproc.tdc_tfine_calibration import one_distribution_fit
from antea.preproc.tdc_tfine_calibration import two_distributions_fit


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
