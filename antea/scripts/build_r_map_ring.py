import sys
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf


### read sensor positions from database
DataSiPM     = db.DataSiPM('petalo', 0)
DataSiPM_idx = DataSiPM.set_index('SensorID')

start     = int(sys.argv[1])
numb      = int(sys.argv[2])
threshold = int(sys.argv[3])

folder = '/folder_path/'
file_full = folder + 'input_file_name.{0:03d}.pet.h5'
evt_file = '/folder_path/output_file_name.{0}_{1}_{2}'.format(start, numb, threshold)

true_r1, true_r2   = [], []
var_phi1, var_phi2 = [], []
var_z1, var_z2     = [], []

touched_sipms1, touched_sipms2 = [], []

for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response = pd.read_hdf(file_name, 'MC/waveforms')
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/waveforms in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    sel_df = rf.find_SiPMs_over_threshold(sns_response, threshold)

    particles = pd.read_hdf(file_name, 'MC/particles')
    hits      = pd.read_hdf(file_name, 'MC/hits')
    events    = particles.event_id.unique()

    for evt in events[:]:

        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits[hits.event_id           == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if not select: continue

        if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511):
            continue

        waveforms = sel_df[sel_df.event_id == evt]
        if len(waveforms) == 0: continue

        q1, q2, pos1, pos2 = rf.assign_sipms_to_gammas(waveforms, true_pos, DataSiPM_idx)

        if len(pos1) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q1)

            pos_z  = np.array(pos1)[:,2]
            mean_z = np.average(pos_z, weights=q1)
            var_z  = np.average((pos_z-mean_z)**2, weights=q1)

            reco_cart = np.average(pos1, weights=q1, axis=0)

            var_phi1.append(var_phi)
            var_z1.append(var_z)
            touched_sipms1.append(len(pos1))

            r = np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2)
            true_r1.append(r)

        else:
            var_phi1.append(1.e9)
            var_z1.append(1.e9)
            touched_sipms1.append(1.e9)
            true_r1.append(1.e9)

        if len(pos2) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q2)

            pos_z  = np.array(pos2)[:,2]
            mean_z = np.average(pos_z, weights=q2)
            var_z  = np.average((pos_z-mean_z)**2, weights=q2)

            reco_cart = np.average(pos2, weights=q2, axis=0)

            var_phi2.append(var_phi)
            var_z2.append(var_z)
            touched_sipms2.append(len(pos2))

            r = np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2)
            true_r2.append(r)

        else:
            var_phi2.append(1.e9)
            var_z2.append(1.e9)
            touched_sipms2.append(1.e9)
            true_r2.append(1.e9)

a_true_r1  = np.array(true_r1)
a_true_r2  = np.array(true_r2)
a_var_phi1 = np.array(var_phi1)
a_var_phi2 = np.array(var_phi2)
a_var_z1   = np.array(var_z1)
a_var_z2   = np.array(var_z2)

a_touched_sipms1 = np.array(touched_sipms1)
a_touched_sipms2 = np.array(touched_sipms2)


np.savez(evt_file, a_true_r1=a_true_r1, a_true_r2=a_true_r2, a_var_phi1=a_var_phi1, a_var_phi2=a_var_phi2, a_var_z1=a_var_z1, a_var_z2=a_var_z2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2)
