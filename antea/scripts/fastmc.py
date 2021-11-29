import sys
import argparse
import pandas   as pd

import antea.database.load_db  as db
import antea.mcsim   .errmat   as errmat
import antea.mcsim   .errmat3d as errmat3d
import antea.mcsim   .fastmc3d as fmc
import antea.io      .mc_io    as mcio


""" To run this script
python fastmc.py input_file output_file matrix_folder
"""

def fastmc(input_file, output_file, table_folder):

    err_r_phot_file    = table_folder + '/errmat_test_r_phot_like.npz'
    err_r_compt_file   = table_folder + '/errmat_test_r_compt_like.npz'
    err_phi_phot_file  = table_folder + '/errmat_test_phi_phot_like.npz'
    err_phi_compt_file = table_folder + '/errmat_test_phi_compt_like.npz'
    err_z_phot_file    = table_folder + '/errmat_test_z_phot_like.npz'
    err_z_compt_file   = table_folder + '/errmat_test_z_compt_like.npz'
    err_t_phot_file    = table_folder + '/errmat_test_t_phot_like.npz'
    err_t_compt_file   = table_folder + '/errmat_test_t_compt_like.npz'

    errmat_r_phot    = errmat  .errmat  (err_r_phot_file   )
    errmat_r_compt   = errmat  .errmat  (err_r_compt_file  )
    errmat_phi_phot  = errmat3d.errmat3d(err_phi_phot_file )
    errmat_phi_compt = errmat3d.errmat3d(err_phi_compt_file)
    errmat_z_phot    = errmat3d.errmat3d(err_z_phot_file   )
    errmat_z_compt   = errmat3d.errmat3d(err_z_compt_file  )
    errmat_t_phot    = errmat  .errmat  (err_t_phot_file   )
    errmat_t_compt   = errmat  .errmat  (err_t_compt_file  )

    energy_threshold = 0.98

    try:
        particles = mcio.load_mcparticles(input_file)
    except:
        print(f'File {input_file} not found!')
        exit()
    hits   = mcio.load_mchits(input_file)
    events = particles.event_id.unique()

    reco = pd.DataFrame(columns=['event_id', 'true_energy',
                                  'true_r1', 'true_phi1', 'true_z1', 'true_t1',
                                  'true_r2', 'true_phi2', 'true_z2', 'true_t2',
                                  'phot_like1', 'phot_like2',
                                  'reco_r1', 'reco_phi1', 'reco_z1', 'reco_t1',
                                  'reco_r2', 'reco_phi2', 'reco_z2', 'reco_t2'])
    for evt in events:
        evt_df = fmc.simulate_reco_event(evt, hits, particles,
                                         errmat_r_phot, errmat_phi_phot,  errmat_z_phot,  errmat_t_phot,
                                         errmat_r_compt, errmat_phi_compt, errmat_z_compt, errmat_t_compt,
                                         energy_threshold, photo_range=1., only_phot=False)
        reco = pd.concat([reco, evt_df])

    store = pd.HDFStore(output_file, "w", complib=str("zlib"), complevel=4)
    store.put('reco', reco, format='table', data_columns=True)
    store.close()


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file' ,   help = "input file name"                )
    parser.add_argument('output_file',   help = "output file name"               )
    parser.add_argument('matrix_folder', help = "path of the error matrix folder")
    return parser.parse_args()

if __name__ == "__main__":

    arguments     = parse_args(sys.argv)
    input_file    = arguments.input_file
    output_file   = arguments.output_file
    matrix_folder = arguments.matrix_folder

    fastmc(input_file, output_file, matrix_folder)
