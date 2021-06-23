import sys
import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.core import system_of_units as units

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

from antea.utils.map_functions import load_map
from antea.io.mc_io import read_sensor_bin_width_from_conf
from antea.io.mc_io import load_mcparticles, load_mchits
from antea.io.mc_io import load_mcsns_response, load_mcTOFsns_response


### read sensor positions from database
#DataSiPM     = db.DataSiPM('petalo', 0) # ring
DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
DataSiPM_idx = DataSiPM.set_index('SensorID')

start   = int(sys.argv[1])
numb    = int(sys.argv[2])
thr_r   = float(sys.argv[3])
thr_phi = float(sys.argv[4])
thr_z   = float(sys.argv[5])
thr_e   = float(sys.argv[6])


folder = 'in_folder_name'
file_full = folder + '/full_body_195cm_center.{0:03d}.pet.h5'
evt_file  = 'out_folder_name/full_body_195cm_center_phot_{0}_{1}_{2}_{3}_{4}_{5}'.format(start, numb, int(thr_r), int(thr_phi), int(thr_z), int(thr_e))

rpos_file = 'table_folder_name/r_table_full_body_195cm_thr{}pes.h5'.format(int(thr_r))

Rpos = load_map(rpos_file,
                 group  = "Radius",
                 node   = "f{}pes200bins".format(int(thr_r)),
                 x_name = "PhiRms",
                 y_name = "Rpos",
                 u_name = "RposUncertainty")

c1 = c2 = c3 = c4 = 0
bad = 0

true_r1, true_phi1, true_z1 = [], [], []
reco_r1, reco_phi1, reco_z1 = [], [], []
true_r2, true_phi2, true_z2 = [], [], []
reco_r2, reco_phi2, reco_z2 = [], [], []

photo_response1, photo_response2 = [], []
first_sipm1, first_sipm2         = [], []
first_time1, first_time2         = [], []
touched_sipms1,  touched_sipms2  = [], []
event_ids = []

for ifile in range(start, start+numb):

    file_name = file_full.format(ifile)
    try:
        sns_response = load_mcsns_response(file_name)
    except ValueError:
        print('File {} not found'.format(file_name))
        continue
    except OSError:
        print('File {} not found'.format(file_name))
        continue
    except KeyError:
        print('No object named MC/sns_response in file {0}'.format(file_name))
        continue
    print('Analyzing file {0}'.format(file_name))

    tof_bin_size = read_sensor_bin_width_from_conf(file_name)

    particles    = load_mcparticles      (file_name)
    hits         = load_mchits           (file_name)
    tof_response = load_mcTOFsns_response(file_name)

    events = particles.event_id.unique()

    for evt in events:

        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if not select: continue

        if (len(true_pos) == 1) & (evt_hits.energy.sum() > 0.511):
            continue

        sns_evt = sns_response[sns_response.event_id == evt]
        evt_tof = tof_response[tof_response.event_id == evt]

        sns_resp_r   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_r)
        sns_resp_phi = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_phi)
        sns_resp_z   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_z)
        sns_resp_e   = rf.find_SiPMs_over_threshold(sns_evt, threshold=thr_e)

        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(sns_resp_r, true_pos, DataSiPM_idx)
        if len(q1) == len(q2) == 0:
            c1 += 1
        r1 = r2 = None
        if len(pos1) > 0:
            pos1_phi    = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            _, var_phi1 = rf.phi_mean_var(pos1_phi, q1)
            r1          = Rpos(np.sqrt(var_phi1)).value
        if len(pos2) > 0:
            pos2_phi = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            _, var_phi2 = rf.phi_mean_var(pos2_phi, q2)
            r2          = Rpos(np.sqrt(var_phi2)).value


        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(sns_resp_phi, true_pos, DataSiPM_idx)
        if len(q1) == len(q2) == 0:
            c2 += 1

        phi1 = phi2 = None
        if len(pos1) > 0:
            reco_cart_pos = np.average(pos1, weights=q1, axis=0)
            phi1 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])
        if len(pos2) > 0:
            reco_cart_pos = np.average(pos2, weights=q2, axis=0)
            phi2 = np.arctan2(reco_cart_pos[1], reco_cart_pos[0])


        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(sns_resp_z, true_pos, DataSiPM_idx)
        if len(q1) == len(q2) == 0:
            c3 += 1

        z1 = z2 = None
        if len(pos1) > 0:
            reco_cart_pos = np.average(pos1, weights=q1, axis=0)
            z1 = reco_cart_pos[2]
        if len(pos2) > 0:
            reco_cart_pos = np.average(pos2, weights=q2, axis=0)
            z2 = reco_cart_pos[2]

        sns1, sns2, _, _, q1, q2 = rf.assign_sipms_to_gammas(sns_resp_e, true_pos, DataSiPM_idx)

        if len(q1) == len(q2) == 0:
            c4 += 1

        ## Calculate time first photoelectron
        sns1 = -np.array(sns1)
        sns2 = -np.array(sns2)
        if r1 and phi1 and z1 and len(q1):
            min_id1, min_t1 = rf.find_first_time_of_sensors(evt_tof, sns1)
        if r2 and phi2 and z2 and len(q2):
            min_id2, min_t2 = rf.find_first_time_of_sensors(evt_tof, sns2)

        both_bad = 0

        if r1 and phi1 and z1 and q1:
            reco_r1        .append(r1)
            reco_phi1      .append(phi1)
            reco_z1        .append(z1)
            true_r1        .append(np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2))
            true_phi1      .append(np.arctan2(true_pos[0][1], true_pos[0][0]))
            true_z1        .append(true_pos[0][2])
            photo_response1.append(sum(q1))
            touched_sipms1 .append(len(q1))
            first_sipm1    .append(min_id1)
            first_time1    .append(min_t1*tof_bin_size/units.ps)
            event_ids      .append(evt)
        else:
            reco_r1        .append(1.e9)
            reco_phi1      .append(1.e9)
            reco_z1        .append(1.e9)
            true_r1        .append(1.e9)
            true_phi1      .append(1.e9)
            true_z1        .append(1.e9)
            photo_response1.append(1.e9)
            touched_sipms1 .append(1.e9)
            first_sipm1    .append(1.e9)
            first_time1    .append(1.e9)
            event_ids      .append(evt)
            both_bad += 1


        if r2 and phi2 and z2 and q2:
            reco_r2        .append(r2)
            reco_phi2      .append(phi2)
            reco_z2        .append(z2)
            true_r2        .append(np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2))
            true_phi2      .append(np.arctan2(true_pos[1][1], true_pos[1][0]))
            true_z2        .append(true_pos[1][2])
            photo_response2.append(sum(q2))
            touched_sipms2 .append(len(q2))
            first_sipm2    .append(min_id2)
            first_time2    .append(min_t2*tof_bin_size/units.ps)
        else:
            reco_r2        .append(1.e9)
            reco_phi2      .append(1.e9)
            reco_z2        .append(1.e9)
            true_r2        .append(1.e9)
            true_phi2      .append(1.e9)
            true_z2        .append(1.e9)
            photo_response2.append(1.e9)
            touched_sipms2 .append(1.e9)
            first_sipm2    .append(1.e9)
            first_time2    .append(1.e9)
            both_bad += 1

        if both_bad == 2:
            bad += 1

a_true_r1   = np.array(true_r1)
a_true_phi1 = np.array(true_phi1)
a_true_z1   = np.array(true_z1)
a_reco_r1   = np.array(reco_r1)
a_reco_phi1 = np.array(reco_phi1)
a_reco_z1   = np.array(reco_z1)
a_sns_response1 = np.array(photo_response1)
a_touched_sipms1 = np.array(touched_sipms1)
a_first_time1 = np.array(first_time1)
a_first_sipm1 = np.array(first_sipm1)

a_true_r2   = np.array(true_r2)
a_true_phi2 = np.array(true_phi2)
a_true_z2   = np.array(true_z2)
a_reco_r2   = np.array(reco_r2)
a_reco_phi2 = np.array(reco_phi2)
a_reco_z2   = np.array(reco_z2)
a_sns_response2 = np.array(photo_response2)
a_touched_sipms2 = np.array(touched_sipms2)
a_first_time2 = np.array(first_time2)
a_first_sipm2 = np.array(first_sipm2)

a_event_ids = np.array(event_ids)

np.savez(evt_file, a_true_r1=a_true_r1, a_true_phi1=a_true_phi1, a_true_z1=a_true_z1, a_true_r2=a_true_r2, a_true_phi2=a_true_phi2, a_true_z2=a_true_z2, a_reco_r1=a_reco_r1, a_reco_phi1=a_reco_phi1, a_reco_z1=a_reco_z1, a_reco_r2=a_reco_r2, a_reco_phi2=a_reco_phi2, a_reco_z2=a_reco_z2,  a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2, a_sns_response1=a_sns_response1, a_sns_response2=a_sns_response2, a_first_time1=a_first_time1, a_first_sipm1=a_first_sipm1, a_first_time2=a_first_time2, a_first_sipm2=a_first_sipm2, a_event_ids=a_event_ids)

print('Not passing threshold r = {}, phi = {}, z = {}, E = {}'.format(c1, c2, c3, c4))
print('Both bad = {}'.format(bad))
