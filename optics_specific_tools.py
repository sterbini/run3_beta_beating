import pymask as pm
import numpy as np
import pandas as pd
import fillingpatterns as fp
import os
# The parts marked by (*) in following need to be
# adapted according to knob definitions

def get_python_parameters(job_row):
    python_parameters= {
        'working_folder' : 'test',
        #'mode' : 'b1_with_bb',
        'mode' : 'b4_from_b2_with_bb',
        'optics_file' : 'opticsfile.20',
        'filling_pattern' : 'filling_scheme_mixed',
        # Tolerances for checks [ip1, ip2, ip5, ip8]
        'tol_beta' : [1e-3, 10e-2, 1e-3, 1e-2],
        'tol_sep' : [1e-6, 1e-6, 1e-6, 1e-6],
        'check_betas_at_ips' : True,
        'check_separations_at_ips' : True,
        'save_intermediate_twiss' : True,
        # tolerance for the sanity check of the flat orbit machine
        'flat_tol' : 1e-6,
        'lumi_levelling_ip15' : True,
        'emittance_um' : 2.5,}

    if job_row is None:
        python_parameters["parent_folder"]=os.getcwd()
        python_parameters['working_folder']=python_parameters['parent_folder']+'/'+python_parameters['working_folder']
		
    if job_row is not None:
        python_parameters['mode']=job_row['mode']
        python_parameters['optics_file']=job_row['optics_file']
        python_parameters['parent_folder']=job_row['parent_folder']
        python_parameters['working_folder']=python_parameters['parent_folder']+'/'+job_row['working_folder']


    patt = fp.FillingPattern.from_json(f'{python_parameters["parent_folder"]}/data/input_{python_parameters["filling_pattern"]}.json')

    python_parameters['filling_pattern_handle']=patt
    python_parameters['filling_pattern_handle'].compute_beam_beam_schedule(n_lr_per_side=20)
    python_parameters['bunch_to_track']=1106

    if python_parameters['mode'][0:2]=='b1':
        python_parameters['bb_schedule_to_track']=python_parameters['filling_pattern_handle'].b1.bb_schedule.loc[python_parameters['bunch_to_track']]
    else:
        python_parameters['bb_schedule_to_track']=python_parameters['filling_pattern_handle'].b2.bb_schedule.loc[python_parameters['bunch_to_track']]

    return python_parameters

def sigma_from_tables(optics_file, emittance_um, filling_scheme, parent_folder):
    if filling_scheme=='filling_scheme_mixed':
        if emittance_um==1.8:
            sigma_df=pd.read_pickle(f'{parent_folder}/data/input_params_2484_1.8.pickle')
        elif emittance_um==2.5:
            sigma_df=pd.read_pickle(f'{parent_folder}/data/input_params_2484_2.5.pickle')
        else:
            assert False
    elif filling_scheme=='filling_scheme_bcms':
        if emittance_um==1.8:
            sigma_df=pd.read_pickle(f'{parent_folder}/data/input_params_2736_1.8.pickle')
        elif emittance_um==2.5:
            sigma_df=pd.read_pickle(f'{parent_folder}/data/input_params_2736_2.5.pickle')
        else:
            assert False
    else:
        assert False
    return np.double(sigma_df[sigma_df['optics_files']==optics_file].iloc[0].sigmaz)

def get_mask_parameters(python_parameters):
    mask_parameters = {
    'par_verbose'              : 1,
    # Beam parameters
    'par_beam_norm_emit'       : python_parameters['emittance_um'],   # [um]
    # it will be redefined later
    'par_beam_sigt'            : 0.075,        # [m]
    'par_beam_sige'            : 1.1e-4,       # [-]
    # it will be redifined by the leveling if done
    'par_beam_npart'           : 1.8e11,       # [-]
    'par_beam_energy_tot'      : 7000,         # [GeV]

    # Settings
    'par_oct_current'          : 350.,         # [A]
    'par_chromaticity'         : 15.,            # [-] (Q':5 for colliding bunches, Q':15 for non-colliding bunches)
    'par_vrf_total'            : 12.,          # [MV]

    # Tunes
    'par_qx0'                  : 62.313,
    'par_qy0'                  : 60.318,


    #*************************#
    # Beam-beam configuration #
    #*************************#

    'par_on_bb_switch'         : 1,
    'par_match_with_bb'        : 0,            # should be off at collision
    'par_b_t_dist'             : 25.,          # bunch separation [ns]
    'par_n_inside_D1'          : 5,            # default value for the number of additionnal parasitic encounters inside D1

    'par_nho_IR1'              : 11,           # number of slices for head-on in IR1 (between 0 and 201)
    'par_nho_IR2'              : 11,           # number of slices for head-on in IR2 (between 0 and 201)
    'par_nho_IR5'              : 11,           # number of slices for head-on in IR5 (between 0 and 201)
    'par_nho_IR8'              : 11,           # number of slices for head-on in IR8 (between 0 and 201)

    #*************************#
    #     Leveling in IP8     #
    #*************************#

    # This variables set the leveled luminosity in IP8 (considered if par_on_collision:1)
    'par_lumi_ip15'            : 2e34,         #[Hz/cm2]
    'par_lumi_ip8'             : 2e33,         #[Hz/cm2]
    'par_fullsep_in_sigmas_ip2': 5,            # in sigmas
    # These variables define the number of Head-On collisions in the 4 IPs
    'par_nco_IP1'              : python_parameters['filling_pattern_handle'].n_coll_ATLAS,
    'par_nco_IP2'              : python_parameters['filling_pattern_handle'].n_coll_ALICE,
    'par_nco_IP5'              : python_parameters['filling_pattern_handle'].n_coll_ATLAS,
    'par_nco_IP8'              : python_parameters['filling_pattern_handle'].n_coll_LHCb,

    #*************************#
    #  Errors and corrections #
    #*************************#

    # Select seed for errors
    'par_myseed'               : 0,

    # Set this flag to correct the errors of D2 in the NLC (warning: for now only correcting b3 of D2, still in development)
    'par_correct_for_D2'       : 0,
    # Set this flag to correct the errors of MCBXF in the NLC (warning: this might be less reproducable in reality, use with care)
    'par_correct_for_MCBX'     : 0,

    'par_on_errors_LHC'        : 0,
    'par_on_errors_MBH'        : 0,
    'par_on_errors_Q5'         : 0,
    'par_on_errors_Q4'         : 0,
    'par_on_errors_D2'         : 0,
    'par_on_errors_D1'         : 0,
    'par_on_errors_IT'         : 0,
    'par_on_errors_MCBRD'      : 0,
    'par_on_errors_MCBXF'      : 0,
    }

    mask_parameters['par_beam_sigt']=sigma_from_tables(python_parameters['optics_file'], mask_parameters['par_beam_norm_emit'], python_parameters['filling_pattern'], python_parameters['parent_folder'])
    return mask_parameters

def get_knob_parameters():
    knob_parameters = {
    #IP specific orbit settings
    # The crossing angle will be redifined in the lumininosity leveling with
    # 0.5*(139.14 -20.43 * np.sqrt(beta_ref) + 196.97 * beta_ref-69.72*beta_ref**(3./2))/np.sqrt(beta_ref)
    'par_x1'                : 150.,
    'par_x5'                : 150.,
    'par_sep1'              : 0,
    'par_sep5'              : 0,
    'par_sep2h'             : 1,
    'par_sep2v'             : 0,
    'par_x2h'               : 0,
    'par_x2v'               : 200,
    'par_sep8h'             : 0,
    'par_sep8v'             : 1,
    'par_x8h'               : -250,
    'par_x8v'               : 0,

    # Dispersion correction knob
    'par_on_disp'              : 0,            # Could be optics-dependent

    # Magnets of the experiments
    'par_on_alice'             : 1,
    'par_on_lhcb'              : 1,

    'par_on_sol_atlas'         : 0,
    'par_on_sol_cms'           : 0,
    'par_on_sol_alice'         : 0,
    }
    return knob_parameters

def build_sequence(mad, beam):

    slicefactor = 8 # for production put slicefactor=8

    pm.make_links(force=True, links_dict={
        'optics_indep_macros.madx': 'tools/optics_indep_macros.madx',
        'macro.madx': ('/afs/cern.ch/eng/lhc/optics/runII/2018/toolkit/macro.madx'),
        'optics_runII': '/afs/cern.ch/eng/lhc/optics/runII',
        'optics_runIII': '/afs/cern.ch/eng/lhc/optics/runIII',})


    mylhcbeam = int(beam)

    mad.input('ver_lhc_run = 3')

    mad.input(f'mylhcbeam = {beam}')
    mad.input('option, -echo,warn, -info;')

    # optics dependent macros
    mad.call('macro.madx')
    mad.input('''
    crossing_save: macro = {
    on_x1_aux=on_x1;on_sep1_aux=on_sep1;on_a1_aux=on_a1;on_o1_aux=on_o1;
    on_x2_aux=on_x2;on_sep2_aux=on_sep2;on_a2_aux=on_a2;on_o2_aux=on_o2; on_oe2_aux=on_oe2;
    on_x5_aux=on_x5;on_sep5_aux=on_sep5;on_a5_aux=on_a5;on_o5_aux=on_o5;
    on_x8_aux=on_x8;on_sep8_aux=on_sep8;on_a8_aux=on_a8;on_o8_aux=on_o8;
    on_x2h_aux=on_x2h;
    on_x2v_aux=on_x2v;
    on_sep2h_aux=on_sep2h;
    on_sep2v_aux=on_sep2v;
    on_x8h_aux=on_x8h;
    on_x8v_aux=on_x8v;
    on_sep8h_aux=on_sep8h;
    on_sep8v_aux=on_sep8v;
    on_disp_aux=on_disp;
    on_alice_aux=on_alice;
    on_lhcb_aux=on_lhcb;
    };

    crossing_disable: macro={
    on_x1=0;on_sep1=0;on_a1=0;on_o1=0;
    on_x2=0;on_sep2=0;on_a2=0;on_o2=0;on_oe2=0;
    on_x5=0;on_sep5=0;on_a5=0;on_o5=0;
    on_x8=0;on_sep8=0;on_a8=0;on_o8=0;
    on_x2h=0;
    on_x2v=0;
    on_sep2h=0;
    on_sep2v=0;
    on_x8h=0;
    on_x8v=0;
    on_sep8h=0;
    on_sep8v=0;
    on_disp=0;
    on_alice=0; on_lhcb=0;
    };

    crossing_restore: macro={
    on_x1=on_x1_aux;on_sep1=on_sep1_aux;on_a1=on_a1_aux;on_o1=on_o1_aux;
    on_x2=on_x2_aux;on_sep2=on_sep2_aux;on_a2=on_a2_aux;on_o2=on_o2_aux; on_oe2=on_oe2_aux;
    on_x5=on_x5_aux;on_sep5=on_sep5_aux;on_a5=on_a5_aux;on_o5=on_o5_aux;
    on_x8=on_x8_aux;on_sep8=on_sep8_aux;on_a8=on_a8_aux;on_o8=on_o8_aux;
    on_x2h=on_x2h_aux;
    on_x2v=on_x2v_aux;
    on_sep2h=on_sep2h_aux;
    on_sep2v=on_sep2v_aux;
    on_x8h=on_x8h_aux;
    on_x8v=on_x8v_aux;
    on_sep8h=on_sep8h_aux;
    on_sep8v=on_sep8v_aux;
    on_disp=on_disp_aux;
    on_alice=on_alice_aux; on_lhcb=on_lhcb_aux;
    };
    ''')
    # optics independent macros
    mad.call('optics_indep_macros.madx')

    assert mylhcbeam in [1, 2, 4], "Invalid mylhcbeam (it should be in [1, 2, 4])"

    if mylhcbeam in [1, 2]:
        mad.call('optics_runII/2018/lhc_as-built.seq')
    else:
        mad.call('optics_runII/2018/lhcb4_as-built.seq')

    # New IR7 MQW layout and cabling
    mad.call('optics_runIII/RunIII_dev/IR7-Run3seqedit.madx')

    # Makethin part
    if slicefactor > 0:
        # the variable in the macro is slicefactor
        mad.input(f'slicefactor={slicefactor};')
        mad.call('optics_runII/2018/toolkit/myslice.madx')
        mad.beam()
        for my_sequence in ['lhcb1','lhcb2']:
            if my_sequence in list(mad.sequence):
                mad.input(f'use, sequence={my_sequence}; makethin, sequence={my_sequence}, style=teapot, makedipedge=true;')
    else:
        warnings.warn('The sequences are not thin!')

    # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    for my_sequence in ['lhcb1','lhcb2']:
        if my_sequence in list(mad.sequence):
            mad.input(f'seqedit, sequence={my_sequence}; flatten; cycle, start=IP3; flatten; endedit;')

def apply_optics(mad, optics_file):
    pm.make_links(force=True, links_dict={
        'optics.madx' : 'optics_runIII/RunIII_dev/2022_V1/PROTON/' + optics_file})
    mad.call('optics.madx')

def check_beta_at_ips_against_madvars(beam, twiss_df, variable_dicts, tol):
    twiss_value_checks=[]
    for iip, ip in enumerate([1,2,5,8]):
        for plane in ['x', 'y']:
            # (*) Adapted based on knob definitions
            twiss_value_checks.append({
                    'element_name': f'ip{ip}:1',
                    'keyword': f'bet{plane}',
                    'varname': f'bet{plane}ip{ip}b{beam}',
                    'tol': tol[iip]})
    pm.check_twiss_against_madvars(twiss_value_checks, twiss_df, variable_dicts)


def set_optics_specific_knobs(mad, mode=None):

    kp=get_knob_parameters()

    mad.set_variables_from_dict(params=kp)

    # Set IP knobs
    mad.globals['on_x1'] = kp['par_x1']
    mad.globals['on_sep1'] = kp['par_sep1']

    mad.globals['on_x2h'] = kp['par_x2h']
    mad.globals['on_x2v'] = kp['par_x2v']
    mad.globals['on_sep2h'] = kp['par_sep2h']
    mad.globals['on_sep2v'] = kp['par_sep2v']

    mad.globals['on_x5'] = kp['par_x5']
    mad.globals['on_sep5'] = kp['par_sep5']

    mad.globals['on_x8h'] = kp['par_x8h']
    mad.globals['on_x8v'] = kp['par_x8v']
    mad.globals['on_sep8h'] = kp['par_sep8h']
    mad.globals['on_sep8v'] = kp['par_sep8v']

    mad.globals['on_disp'] = kp['par_on_disp']

    # A check
    if mad.globals.nrj < 500:
        assert kp['par_on_disp'] == 0

    # Spectrometers at experiments
    if kp['par_on_alice'] == 1:
        mad.globals.on_alice = 7000./mad.globals.nrj
    if kp['par_on_lhcb'] == 1:
        mad.globals.on_lhcb = 7000./mad.globals.nrj

    # Solenoids at experiments
    mad.globals.on_sol_atlas = kp['par_on_sol_atlas']
    mad.globals.on_sol_cms = kp['par_on_sol_cms']
    mad.globals.on_sol_alice = kp['par_on_sol_alice']

def check_separations_at_ips_against_madvars(twiss_df_b1, twiss_df_b2,
        variables_dict, tol):

    separations_to_check = []
    for iip, ip in enumerate([2,8]):
        for plane in ['x', 'y']:
            # (*) Adapet based on knob definitions
            separations_to_check.append({
                    'element_name': f'ip{ip}:1',
                    'scale_factor': -2*1e-3,
                    'plane': plane,
                    # knobs like on_sep1h, onsep8v etc
                    'varname': f'on_sep{ip}'+{'x':'h', 'y':'v'}[plane],
                    'tol': tol[iip]})
    separations_to_check.append({ # IP1
            'element_name': f'ip1:1',
                    'scale_factor': -2*1e-3,
                    'plane': 'x',
                    'varname': 'on_sep1',
                    'tol': tol[0]})
    separations_to_check.append({ # IP5
            'element_name': f'ip5:1',
                    'scale_factor': -2*1e-3,
                    'plane': 'y',
                    'varname': 'on_sep5',
                    'tol': tol[2]})
    pm.check_separations_against_madvars(separations_to_check,
            twiss_df_b1, twiss_df_b2, variables_dict)

def twiss_and_check(mad, sequences_to_check, twiss_fname,
        tol_beta=1e-3, tol_sep=1e-6, save_twiss_files=True,
        check_betas_at_ips=True, check_separations_at_ips=True, remove_drifts=True, columns_to_save=['name', 's',
            'keyword', 'parent', 'x', 'y', 'px', 'py', 'dx', 'dy', 'dpx', 'dpy', 'betx', 'bety', 'alfx', 'alfy', 
            'mux','muy', 'k0l', 'k1l', 'k2l', 'k3l', 'k0sl', 'k1sl', 'k2sl', 'k3sl']):
    var_dict = mad.get_variables_dicts()
    twiss_dfs = {}
    summ_dfs = {}
    for ss in sequences_to_check:
        mad.use(ss)
        mad.twiss()
        tdf = mad.get_twiss_df('twiss')
        twiss_dfs[ss] = tdf
        sdf = mad.get_summ_df('summ')
        summ_dfs[ss] = sdf

    if save_twiss_files:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            try:
                tt=tt[columns_to_save]
            except:
                print("Check the columns_to_save, there was a problem...")
            if remove_drifts:
                tt=tt[tt['keyword']!='drift']
            if twiss_fname is not None:
                tt.to_parquet(twiss_fname + f'_seq_{ss}.parquet')

    if check_betas_at_ips:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            check_beta_at_ips_against_madvars(beam=ss[-1],
                    twiss_df=tt,
                    variable_dicts=var_dict,
                    tol=tol_beta)
        print('IP beta test against knobs passed!')

    if check_separations_at_ips:
        twiss_df_b1 = twiss_dfs['lhcb1']
        twiss_df_b2 = twiss_dfs['lhcb2']
        check_separations_at_ips_against_madvars(twiss_df_b1, twiss_df_b2,
                var_dict, tol=tol_sep)
        print('IP separation test against knobs passed!')

    other_data = {}
    other_data.update(var_dict)
    other_data['summ_dfs'] = summ_dfs

    return twiss_dfs, other_data

def twiss_with_no_use(mad, sequences_to_check, twiss_fname,
        tol_beta=1e-3, tol_sep=1e-6, save_twiss_files=True,
        check_betas_at_ips=True, check_separations_at_ips=True, remove_drifts=True, columns_to_save=['name', 's', 
            'keyword', 'parent', 'x', 'y', 'px', 'py', 'dx', 'dy', 'dpx', 'dpy', 'betx', 'bety', 'alfx', 'alfy', 
            'mux','muy', 'k0l', 'k1l', 'k2l', 'k3l', 'k0sl', 'k1sl', 'k2sl', 'k3sl']):

    var_dict = mad.get_variables_dicts()
    twiss_dfs = {}
    summ_dfs = {}
    for ss in sequences_to_check:
        mad.twiss()
        tdf = mad.get_twiss_df('twiss')
        twiss_dfs[ss] = tdf
        sdf = mad.get_summ_df('summ')
        summ_dfs[ss] = sdf

    if save_twiss_files:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            try:
                tt=tt[columns_to_save]
            except:
                print("Check the columns_to_save, there was a problem...")
            if remove_drifts:
                tt=tt[tt['keyword']!='drift']
            summary_df = summ_dfs[ss]
            if twiss_fname is not None:
                tt.to_parquet(twiss_fname + f'_seq_{ss}.parquet')
                summary_df.to_parquet(twiss_fname + f'_summary_{ss}.parquet')

    if check_betas_at_ips:
        for ss in sequences_to_check:
            tt = twiss_dfs[ss]
            check_beta_at_ips_against_madvars(beam=ss[-1],
                    twiss_df=tt,
                    variable_dicts=var_dict,
                    tol=tol_beta)
        print('IP beta test against knobs passed!')

    if check_separations_at_ips:
        twiss_df_b1 = twiss_dfs['lhcb1']
        twiss_df_b2 = twiss_dfs['lhcb2']
        check_separations_at_ips_against_madvars(twiss_df_b1, twiss_df_b2,
                var_dict, tol=tol_sep)
        print('IP separation test against knobs passed!')

    other_data = {}
    other_data.update(var_dict)
    other_data['summ_dfs'] = summ_dfs

    return twiss_dfs, other_data

def filter_bb_df(bb_df, bb_schedule_to_track):
    if bb_schedule_to_track['collides in ATLAS/CMS']==False:
        bb_df=bb_df[~((bb_df['ip_name']=='ip1') & (bb_df['label']=='bb_ho'))]
        bb_df=bb_df[~((bb_df['ip_name']=='ip5') & (bb_df['label']=='bb_ho'))]
    if bb_schedule_to_track['collides in LHCB']==False:
        bb_df=bb_df[~((bb_df['ip_name']=='ip8') & (bb_df['label']=='bb_ho'))]
    if bb_schedule_to_track['collides in ALICE']==False:
        bb_df=bb_df[~((bb_df['ip_name']=='ip2') & (bb_df['label']=='bb_ho'))]

    bb_df_ho=bb_df[bb_df['label']=='bb_ho'].copy()

    # BBLR ATLAS
    lr_ATLAS=bb_df[(bb_df['ip_name']=='ip1') & (bb_df['label']=='bb_lr')].copy()
    lr_ATLAS=lr_ATLAS[lr_ATLAS['identifier'].apply(lambda x: x in bb_schedule_to_track['Positions in ATLAS/CMS'])]

    # BBLR ALICE
    lr_ALICE=bb_df[(bb_df['ip_name']=='ip2') & (bb_df['label']=='bb_lr')].copy()
    lr_ALICE=lr_ALICE[lr_ALICE['identifier'].apply(lambda x: x in bb_schedule_to_track['Positions in ALICE'])]

    # BBLR CMS
    lr_CMS=bb_df[(bb_df['ip_name']=='ip5') & (bb_df['label']=='bb_lr')].copy()
    lr_CMS=lr_CMS[lr_CMS['identifier'].apply(lambda x: x in bb_schedule_to_track['Positions in ATLAS/CMS'])]

    # BBLR LHCB
    lr_LHCB=bb_df[(bb_df['ip_name']=='ip8') & (bb_df['label']=='bb_lr')].copy()
    lr_LHCB=lr_LHCB[lr_LHCB['identifier'].apply(lambda x: x in bb_schedule_to_track['Positions in LHCB'])]

    aux=pd.concat([bb_df_ho, lr_ATLAS, lr_ALICE, lr_CMS, lr_LHCB])
    return aux
