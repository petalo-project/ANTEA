import pandas as pd
import numpy  as np
import os

from antea.preproc.qdc_corrections import correct_efine_wrap_around
from antea.preproc.qdc_corrections import apply_qdc_autocorrection
from antea.preproc.qdc_corrections import compute_qdc_calibration_using_mode
from antea.preproc.qdc_corrections import create_qdc_interpolator_df
from antea.preproc.qdc_corrections import compute_efine_correction_using_linear_interpolation


def test_correct_efine_wrap_around():
    ''' Check the efine correction '''
    df              = pd.DataFrame({'efine':[1,20,300,450,1020]})
    expected_result = np.array([15, 34, 314, 464, 10])

    correct_efine_wrap_around(df)

    np.testing.assert_array_equal(df['efine'].values, expected_result)


def test_apply_qdc_autocorrection():
    '''
    Check the result of the efine  correction due to the offset
    '''
    df     = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                           'channel_id': [4, 10, 13, 25, 29, 36, 42, 47, 58, 63, 42, 58],
                           'tac_id'    : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0],
                           'intg_w'    : [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 600, 800],
                           'efine'     : [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]})


    df_qdc = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                           'channel_id': [4,10, 13, 25, 29, 36, 42, 47, 58, 63 ],
                           'tac_id'    : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                           'intg_w'    : [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                           'offset'    : [0, 5, 30, 70, 100, 20, 200, 260, 300, 410]})

    diff            = apply_qdc_autocorrection(df, df_qdc).values
    expected_result = np.array([0, 45, 70, 80, 100, 230, 100, 300, 90, 100, 250, 40])

    np.testing.assert_array_equal(diff, expected_result)


def test_compute_qdc_calibration_using_mode():
    '''
    Check the data ordering in function of tofpet_id and channel_id and
    the addition of integration window and efine corrected.
    '''
    df_tpulse   = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'channel_id': [14, 17, 15, 17, 14, 15, 15, 14, 17, 14],
                                'tac_id'    : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                                'efine'     : [50, 100, 125, 300, 200, 350, 225, 290, 360, 150],
                                'ecoarse'   : [150, 582, 6, 700, 1000, 1024, 300, 250, 800, 525],
                                'tcoarse'   : [100, 4387, 28675, 33400, 40800, 59000, 200, 1024,  3600, 40200]})

    df_calib    = compute_qdc_calibration_using_mode(df_tpulse)
    df_expected = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                'channel_id': [14, 14, 14, 14, 15, 15, 15, 17, 17, 17],
                                'tac_id'    : [0, 0, 1, 3, 1, 2, 2, 0, 1, 3],
                                'intg_w'    : [50, 136, 261, 250, 392, 0, 100, 272, 2000, 68],
                                'efine'     : [64, 214, 164, 304, 364, 0, 239, 374, 114, 314]})

    np.testing.assert_array_equal(df_calib, df_expected)



def test_create_qdc_interpolator_df(output_tmpdir):
    '''
    Check the qdc interpolation result
    '''
    fname_qdc_0 = os.path.join(output_tmpdir, 'test_qdc_0.h5')
    fname_qdc_2 = os.path.join(output_tmpdir, 'test_qdc_2.h5')

    df_qdc_0 = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'channel_id': [16, 16, 16, 16, 16, 17, 17, 17, 17, 17],
                             'tac_id'    : [0, 0, 1, 1, 1, 0, 0, 2, 2, 2],
                             'intg_w'    : [10, 20, 60, 70, 80, 300, 400, 100, 120, 140],
                             'efine'     : [20, 40, 100, 110, 120, 300, 400, 200, 220, 240]})

    df_qdc_2 = pd.DataFrame({'tofpet_id' : [1, 1, 1, 1, 1, 1, 1],
                             'channel_id': [29, 29, 33, 33, 33, 57, 57],
                             'tac_id'    : [3, 3, 0, 0, 0, 1, 1],
                             'intg_w'    : [100, 125, 200, 210, 220, 300, 350],
                             'efine'     : [50, 70, 110, 120, 130, 240, 280]})

    df_qdc_0.to_hdf (fname_qdc_0, key = 'data', format = 'table')
    df_qdc_2.to_hdf (fname_qdc_2, key = 'data', format = 'table')

    df_0 = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0],
                         'channel_id' : [16, 16, 17, 17],
                         'tac_id'     : [0, 1, 0, 2],
                         'intg_w'     : [15, 73, 350, 130]})

    df_2 = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0, 1, 1, 1],
                         'channel_id' : [16, 16, 17, 17, 29, 33, 57],
                         'tac_id'     : [0, 1, 0, 2, 3, 0, 1],
                         'intg_w'     : [15, 73, 350, 130, 115, 213, 325]})

    df_interp_0 = create_qdc_interpolator_df(fname_qdc_0)
    df_0['correction'] = df_0.apply(lambda row: df_interp_0[row.tofpet_id, row.channel_id,
                                                           row.tac_id](row.intg_w), axis=1).astype(np.float64) #df.correction is object
    df_expected_0      = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0],
                                       'channel_id' : [16, 16, 17, 17],
                                       'tac_id'     : [0, 1, 0, 2],
                                       'intg_w'     : [15, 73, 350, 130],
                                       'correction' : [30, 113, 350, 230]})



    df_interp_2 = create_qdc_interpolator_df(fname_qdc_0, fname_qdc_2)
    df_2['correction'] = df_2.apply(lambda row: df_interp_2[row.tofpet_id, row.channel_id,
                                                            row.tac_id](row.intg_w), axis=1).astype(np.float64) #df.correction is object
    df_expected_2      = pd.DataFrame({'tofpet_id'  : [0, 0, 0, 0, 1, 1, 1],
                                       'channel_id' : [16, 16, 17, 17, 29, 33, 57],
                                       'tac_id'     : [0, 1, 0, 2, 3, 0, 1],
                                       'intg_w'     : [15, 73, 350, 130, 115, 213, 325],
                                       'correction' : [30, 113, 350, 230, 62, 123, 260]})


    np.testing.assert_array_equal(df_0, df_expected_0)
    np.testing.assert_array_equal(df_2, df_expected_2)


def test_compute_efine_correction_using_linear_interpolation(output_tmpdir):
    '''
    Check the corrected efine after using linear interpolation for bias substraction
    '''
    fname_qdc = os.path.join(output_tmpdir, 'test_0.h5')

    df_qdc    = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              'channel_id': [16, 16, 16, 16, 16, 17, 17, 17, 17, 17],
                              'tac_id'    : [0, 0, 1, 1, 1, 0, 0, 2, 2, 2],
                              'intg_w'    : [10, 20, 60, 70, 80, 300, 400, 100, 120, 140],
                              'efine'     : [20, 40, 100, 110, 120, 300, 400, 200, 220, 240]})

    df_qdc.to_hdf (fname_qdc, key = 'data', format = 'table')

    df        = pd.DataFrame({'tofpet_id' : [0, 0, 0, 0],
                              'channel_id': [16, 16, 17, 17],
                              'tac_id'    : [0, 1, 0, 2],
                              'intg_w'    : [15, 73, 350, 130],
                              'efine'     : [130, 223, 490, 460]})

    df_interpolators = create_qdc_interpolator_df(fname_qdc)
    compute_efine_correction_using_linear_interpolation(df, df_interpolators)

    df_expected = pd.DataFrame({'tofpet_id'      : [0, 0, 0, 0],
                                'channel_id'     : [16, 16, 17, 17],
                                'tac_id'         : [0, 1, 0, 2],
                                'intg_w'         : [15, 73, 350, 130],
                                'efine'          : [130, 223, 490, 460],
                                'efine_corrected':[100, 110, 140, 230]})
    df_expected['efine_corrected'] = df_expected['efine_corrected'].astype(np.float64)

    np.testing.assert_array_equal(df, df_expected)
