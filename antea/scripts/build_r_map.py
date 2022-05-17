import sys
import numpy  as np
import pandas as pd

import antea.database.load_db      as db
import antea.reco.reco_functions   as rf
import antea.reco.mctrue_functions as mcf

""" To run this script:
python build_r_map.py input_file output_file threshold
where:
-  input_file is the nexus file used to extract
the radial dependence.
- output_file is an npz file with the information
needed to build the R map.
- threshold is the charge above which a SiPM is
taken into account to extract that information.
"""

### read sensor positions from database
#DataSiPM     = db.DataSiPM('petalo', 0) # ring

def build_r_map(input_file: str, output_file: str, threshold: float):

    """
    This function extracts the true radial position
    and the variance of the SiPM positions
    in phi and z for each gamma interaction.
    """

    DataSiPM     = db.DataSiPMsim_only('petalo', 0) # full body PET
    DataSiPM_idx = DataSiPM.set_index('SensorID')

    try:
        sns_response = pd.read_hdf(input_file, 'MC/sns_response')
    except ValueError:
        print(f'File {input_file} not found')
        exit()
    except OSError:
        print(f'File {input_file} not found')
        exit()
    except KeyError:
        print(f'No object named MC/sns_response in file {input_file}')
        exit()
    print(f'Analyzing file {input_file}')

    sel_df = sns_response[sns_response.charge > threshold]

    particles = pd.read_hdf(input_file, 'MC/particles')
    hits      = pd.read_hdf(input_file, 'MC/hits')
    events    = particles.event_id.unique()

    true_r1, true_r2   = [], []
    var_phi1, var_phi2 = [], []
    var_z1, var_z2     = [], []

    touched_sipms1, touched_sipms2 = [], []

    for evt in events:

        ### Select photoelectric events only
        evt_parts = particles[particles.event_id == evt]
        evt_hits  = hits     [hits     .event_id == evt]
        select, true_pos = mcf.select_photoelectric(evt_parts, evt_hits)
        if not select: continue

        sns_resp = sel_df[sel_df.event_id == evt]
        if len(sns_resp) == 0: continue

        _, _, pos1, pos2, q1, q2 = rf.assign_sipms_to_gammas(sns_resp, true_pos, DataSiPM_idx)

        if len(pos1) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos1))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q1)

            pos_z  = np.array(pos1)[:,2]
            mean_z = np.average(pos_z, weights=q1)
            var_z  = np.average((pos_z-mean_z)**2, weights=q1)
            r      = np.sqrt(true_pos[0][0]**2 + true_pos[0][1]**2)

            var_phi1      .append(var_phi)
            var_z1        .append(var_z)
            touched_sipms1.append(len(pos1))
            true_r1       .append(r)

        else:
            var_phi1      .append(1.e9)
            var_z1        .append(1.e9)
            touched_sipms1.append(1.e9)
            true_r1       .append(1.e9)

        if len(pos2) > 0:
            pos_phi    = rf.from_cartesian_to_cyl(np.array(pos2))[:,1]
            _, var_phi = rf.phi_mean_var(pos_phi, q2)

            pos_z  = np.array(pos2)[:,2]
            mean_z = np.average(pos_z, weights=q2)
            var_z  = np.average((pos_z-mean_z)**2, weights=q2)
            r      = np.sqrt(true_pos[1][0]**2 + true_pos[1][1]**2)

            var_phi2      .append(var_phi)
            var_z2        .append(var_z)
            touched_sipms2.append(len(pos2))
            true_r2       .append(r)

        else:
            var_phi2      .append(1.e9)
            var_z2        .append(1.e9)
            touched_sipms2.append(1.e9)
            true_r2       .append(1.e9)

    a_true_r1  = np.array(true_r1)
    a_true_r2  = np.array(true_r2)
    a_var_phi1 = np.array(var_phi1)
    a_var_phi2 = np.array(var_phi2)
    a_var_z1   = np.array(var_z1)
    a_var_z2   = np.array(var_z2)

    a_touched_sipms1 = np.array(touched_sipms1)
    a_touched_sipms2 = np.array(touched_sipms2)


    np.savez(output_file, a_true_r1=a_true_r1, a_true_r2=a_true_r2, a_var_phi1=a_var_phi1, a_var_phi2=a_var_phi2, a_var_z1=a_var_z1, a_var_z2=a_var_z2, a_touched_sipms1=a_touched_sipms1, a_touched_sipms2=a_touched_sipms2)

if __name__ == "__main__":

    input_file  = str(sys.argv[1])
    output_file = str(sys.argv[2])
    threshold   = int(sys.argv[3])
    build_r_map(input_file, output_file, threshold)