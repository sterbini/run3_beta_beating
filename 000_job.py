import sys, os
import pickle
import numpy as np
import pandas as pd

if len(sys.argv) > 1:
    job_df = pd.read_pickle(sys.argv[1])
    job_index = int(sys.argv[2])
    job_row = job_df.iloc[job_index].copy()
else:
    job_row = None

# Import pymask
sys.path.append('/afs/cern.ch/eng/tracking-tools/modules')
import pymask as pm
from pymask import luminosity as lumi 
import optics_specific_tools as ost


python_parameters={'parent_folder': os.getcwd()}
sys.path.append(python_parameters['parent_folder'])# to read local data
python_parameters=ost.get_python_parameters(python_parameters,job_row)
mask_parameters=ost.get_mask_parameters(python_parameters)
knob_parameters=ost.get_knob_parameters()

directory=python_parameters['working_folder']
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

Madx = pm.Madxp

# Select mode
mode = python_parameters['mode']
# mode can be
# 'b1_with_bb'
# 'b1_with_bb_legacy_macros'
# 'b1_without_bb'
# 'b4_without_bb'
# 'b4_from_b2_without_bb'
# 'b4_from_b2_with_bb'

# Tolerances for checks [ip1, ip2, ip5, ip8]
tol_beta =python_parameters['tol_beta']
tol_sep = python_parameters['tol_sep']

pm.make_links(force=True, links_dict={
    'tracking_tools': '/afs/cern.ch/eng/tracking-tools',
    'modules': 'tracking_tools/modules',
    'tools': 'tracking_tools/tools',
    'beambeam_macros': 'tracking_tools/beambeam_macros',
    'errors': 'tracking_tools/errors'})

optics_file = python_parameters['optics_file']

check_betas_at_ips = python_parameters['check_betas_at_ips']
check_separations_at_ips = python_parameters['check_separations_at_ips']
save_intermediate_twiss = python_parameters['save_intermediate_twiss']

# Check and load parameters 
pm.checks_on_parameter_dict(mask_parameters)

# Define configuration
(beam_to_configure, sequences_to_check, sequence_to_track, generate_b4_from_b2,
    track_from_b4_mad_instance, enable_bb_python, enable_bb_legacy,
    force_disable_check_separations_at_ips,
    ) = pm.get_pymask_configuration(mode)
if force_disable_check_separations_at_ips:
    check_separations_at_ips = False

# Start mad
mad = Madx()

# Build sequence
ost.build_sequence(mad, beam=beam_to_configure)

# Apply optics
ost.apply_optics(mad, optics_file=optics_file)

# Force disable beam-beam when needed
if not(enable_bb_legacy) and not(enable_bb_python):
    mask_parameters['par_on_bb_switch'] = 0.

# Pass parameters to mad
mad.set_variables_from_dict(params=mask_parameters)

# Prepare sequences and attach beam
mad.call("modules/submodule_01a_preparation.madx")
mad.call("modules/submodule_01b_beam.madx")

# Test machine before any change
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_from_optics',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips,
        check_separations_at_ips=check_separations_at_ips)

# Set phase, apply and save crossing
mad.call("modules/submodule_01c_phase.madx")

# Set optics-specific knobs
ost.set_optics_specific_knobs(mad, mode)

# Crossing-save and some reference measurements
mad.input('exec, crossing_save')
mad.call("modules/submodule_01e_final.madx")

# Test flat machine
mad.input('exec, crossing_disable')
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_no_crossing',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=check_separations_at_ips)
# Check flatness
flat_tol = python_parameters['flat_tol']
for ss in twiss_dfs.keys():
    tt = twiss_dfs[ss]
    assert np.max(np.abs(tt.x)) < flat_tol
    assert np.max(np.abs(tt.y)) < flat_tol

# Check machine after crossing restore
mad.input('exec, crossing_restore')
twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_with_crossing',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=check_separations_at_ips)

mad.use(f'lhcb{beam_to_configure}')

# # Call leveling module
# if mode=='b4_without_bb':
#     print('Leveling not working in this mode!')
# else:
#     mad.call("modules/module_02_lumilevel.madx")

python_parameters['beta_ref']=mad.globals['betx_IP1']

if enable_bb_legacy or mode=='b4_without_bb':
    mad.use(f'lhcb{beam_to_configure}')
    if mode=='b4_without_bb':
        print('Leveling not working in this mode!')
    else:
        # Luminosity levelling
        print('Luminosities before leveling (crab cavities are not considered):')
        lumi.print_luminosity(mad, twiss_dfs,
                mask_parameters['par_nco_IP1'], mask_parameters['par_nco_IP2'],
                mask_parameters['par_nco_IP5'], mask_parameters['par_nco_IP8'])

        mad.call("modules/module_02_lumilevel.madx")

        print('Luminosities after leveling:')
        lumi.print_luminosity(mad, twiss_dfs,
                mask_parameters['par_nco_IP1'], mask_parameters['par_nco_IP2'],
                mask_parameters['par_nco_IP5'], mask_parameters['par_nco_IP8'])
else:
    from scipy.optimize import least_squares

    print('Luminosities before leveling:')
    lumi.print_luminosity(mad, twiss_dfs,
            mask_parameters['par_nco_IP1'], mask_parameters['par_nco_IP2'],
            mask_parameters['par_nco_IP5'], mask_parameters['par_nco_IP8'])

    L_target=mask_parameters['par_lumi_ip15']
    starting_guess=mask_parameters['par_beam_npart']
    beta_ref=mad.globals['betx_IP1']

    crossing_angle=0.5*(139.14 -20.43 * np.sqrt(beta_ref) + 196.97 * beta_ref-69.72*beta_ref**(3./2))/np.sqrt(beta_ref)
    def function_to_minimize_IP15(n_part):
        my_dict_IP1=lumi.get_luminosity_dict(mad, twiss_dfs,'ip1', mask_parameters['par_nco_IP1'])  
        my_dict_IP1['N1']=n_part
        my_dict_IP1['N2']=n_part
        my_dict_IP1['py_1']=crossing_angle*1e-6
        my_dict_IP1['py_2']=-crossing_angle*1e-6
        my_dict_IP5=lumi.get_luminosity_dict(mad, twiss_dfs, 'ip5', mask_parameters['par_nco_IP5'])  
        my_dict_IP5['N1']=n_part
        my_dict_IP5['N2']=n_part
        my_dict_IP5['px_1']=crossing_angle*1e-6
        my_dict_IP5['px_2']=-crossing_angle*1e-6
        return lumi.L(**my_dict_IP1)+lumi.L(**my_dict_IP5)-2*L_target
    
    if python_parameters['lumi_levelling_ip15']:
        aux=least_squares(function_to_minimize_IP15, starting_guess)
        mad.sequence.lhcb1.beam.npart=aux['x'][0]
        mad.sequence.lhcb2.beam.npart=aux['x'][0]
        mask_parameters['par_beam_npart']=aux['x'][0]
    
    knob_parameters['par_x1']=crossing_angle
    knob_parameters['par_x5']=crossing_angle
    mad.globals['on_x1']=crossing_angle 
    mad.globals['on_x5']=crossing_angle 


    # Leveling in IP8
    L_target_ip8 = mask_parameters['par_lumi_ip8']
    def function_to_minimize_ip8(sep8v_m):
        my_dict_IP8=lumi.get_luminosity_dict(
            mad, twiss_dfs, 'ip8', mask_parameters['par_nco_IP8'])
        my_dict_IP8['y_1']=np.abs(sep8v_m)
        my_dict_IP8['y_2']=-np.abs(sep8v_m)
        return np.abs(lumi.L(**my_dict_IP8) - L_target_ip8)
    sigma_x_b1_ip8=np.sqrt(twiss_dfs['lhcb1'].loc['ip8:1'].betx*mad.sequence.lhcb1.beam.ex)
    optres_ip8=least_squares(function_to_minimize_ip8, sigma_x_b1_ip8)
    mad.globals['on_sep8v'] = np.sign(mad.globals['on_sep8v']) * np.abs(optres_ip8['x'][0])*1e3

    # Halo collision in IP2
    sigma_y_b1_ip2=np.sqrt(twiss_dfs['lhcb1'].loc['ip2:1'].bety*mad.sequence.lhcb1.beam.ey)
    mad.globals['on_sep2h']=np.sign(mad.globals['on_sep2h'])*mask_parameters['par_fullsep_in_sigmas_ip2']*sigma_y_b1_ip2/2*1e3

    # Re-save knobs
    mad.input('exec, crossing_save')

    twiss_dfs, other_data = ost.twiss_and_check(mad, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_after_leveling',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=check_separations_at_ips)
    
    print('Luminosities after leveling:')
    lumi.print_luminosity(mad, twiss_dfs,
            mask_parameters['par_nco_IP1'], mask_parameters['par_nco_IP2'],
            mask_parameters['par_nco_IP5'], mask_parameters['par_nco_IP8'])
    
    import pandas as pd
    for ii in [1,2,5,8]:
        pd.DataFrame([lumi.get_luminosity_dict(mad, twiss_dfs, f'ip{ii}', mask_parameters[f'par_nco_IP{ii}'])]).to_parquet(f'lumi_dict_ip{ii}.parquet')
        python_parameters[f'L_IP{ii}']=lumi.L(**lumi.get_luminosity_dict(mad, twiss_dfs, f'ip{ii}', mask_parameters[f'par_nco_IP{ii}']))

mad.input('on_disp = 0')

# Prepare bb dataframes
if enable_bb_python:
    bb_dfs = pm.generate_bb_dataframes(mad,
        ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
        harmonic_number=35640,
        numberOfLRPerIRSide=[25, 20, 25, 20],
        bunch_spacing_buckets=10,
        numberOfHOSlices=11,
        bunch_population_ppb=None,
        sigmaz_m=None,
        z_crab_twiss = 0,
        remove_dummy_lenses=True)

if mode=='b1_with_bb':
    bb_dfs['b1']=ost.filter_bb_df(bb_dfs['b1'], python_parameters['bb_schedule_to_track'])

if mode=='b4_from_b2_with_bb':
    bb_dfs['b4']=ost.filter_bb_df(bb_dfs['b4'], python_parameters['bb_schedule_to_track'])

# Here the datafremes can be edited, e.g. to set bbb intensity

# Generate mad instance for b4
if generate_b4_from_b2:
    mad_b4 = Madx()
    ost.build_sequence(mad_b4, beam=4)
    ost.apply_optics(mad_b4, optics_file=optics_file)

    pm.configure_b4_from_b2(mad_b4, mad)

    twiss_dfs_b2, other_data_b2 = ost.twiss_and_check(mad,
            sequences_to_check=['lhcb2'],
            tol_beta=tol_beta, tol_sep=tol_sep,
            twiss_fname='twiss_b2_for_b4check',
            save_twiss_files=save_intermediate_twiss,
            check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)

    twiss_dfs_b4, other_data_b4 = ost.twiss_and_check(mad_b4,
            sequences_to_check=['lhcb2'],
            tol_beta=tol_beta, tol_sep=tol_sep,
            twiss_fname='twiss_b4_for_b4check',
            save_twiss_files=save_intermediate_twiss,
            check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)



# We working exclusively on the sequence to track
# Select mad object
if track_from_b4_mad_instance:
    mad_track = mad_b4
else:
    mad_track = mad

mad_collider = mad
del(mad)

# Twiss machine to track
twiss_dfs, other_data = ost.twiss_and_check(mad_track, sequences_to_check,
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_track_intermediate',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=check_betas_at_ips, check_separations_at_ips=False)


# Install bb lenses
if enable_bb_python:
    if track_from_b4_mad_instance:
        bb_df_track = bb_dfs['b4']
        assert(sequence_to_track=='lhcb2')
    else:
        bb_df_track = bb_dfs['b1']
        assert(sequence_to_track=='lhcb1')

    pm.install_lenses_in_sequence(mad_track, bb_df_track, sequence_to_track)

    # Disable bb
    mad_track.globals.on_bb_charge = 0
else:
    bb_df_track = None


# Legacy bb macros
if enable_bb_legacy:
    assert(beam_to_configure == 1)
    assert(not(track_from_b4_mad_instance))
    assert(not(enable_bb_python))
    mad_track.call("modules/module_03_beambeam.madx")

# # Install crab cavities
# mad_track.call("tools/enable_crabcavities.madx")

# # Install crab cavities
#mad_track.call("modules/submodule_04_1a_install_crabs.madx")

# Save references (orbit at IPs)
mad_track.call('modules/submodule_04_1b_save_references.madx')

# Switch off dipersion correction knob
mad_track.globals.on_disp = 0.

# Final use
mad_track.use(sequence_to_track)
# Disable use
mad_track._use = mad_track.use
mad_track.use = None

# # Install and correct errors
# mad_track.call("modules/module_04_errors.madx")

# Machine tuning (enables bb)
#mad_track.call("modules/module_05_tuning.madx")

mad_track.call("modules/submodule_05a_MO.madx")
mad_track.call("modules/submodule_05b_coupling.madx")
mad_track.call("modules/submodule_05c_limit.madx")
#mad_track.call("modules/submodule_05d_matching.madx")

def matching_Q_Qp(mad,q_knob_1,q_knob_2,qp_knob_1,qp_knob_2):
    mad.input(f'''
print, text="";
print, text="";
print, text="  --- Submodule 5d: matching";
print, text="  --------------------------";
print, text="";
print, text="";



!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!                 matching of orbit, tune and chromaticity
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if(par_match_with_bb==1) {{ON_BB_CHARGE:=1;}};    ! W/o head-on Q and Q' are matched with bb

!Rematch the Xscheme towards specified separation and Xangle in IP1/2/5/8
call, file="tools/rematchCOIP.madx";
!Rematch the CO in the arc for dispersion correction
if(on_disp<>0) {{call, file="tools/rematchCOarc.madx";}};

kqtf={q_knob_1};kqtd={q_knob_2};{q_knob_1}:=kqtf;{q_knob_2}:=kqtd;
ksf={qp_knob_1};ksd={qp_knob_2};{qp_knob_1}:=ksf;{qp_knob_2}:=ksd;

match;
global, q1=qx0, q2=qy0;
vary,   name=kqtf, step=1.0E-7 ;
vary,   name=kqtd, step=1.0E-7 ;
lmdif,  calls=100, tolerance=1.0E-21;
endmatch;

match,chrom;
global, dq1=qprime, dq2=qprime;
vary,   name=ksf;
vary,   name=ksd;
lmdif,  calls=100, tolerance=1.0E-21;
endmatch;

match,chrom;
global, dq1=qprime, dq2=qprime;
global, q1=qx0, q2=qy0;
vary,   name=ksf;
vary,   name=ksd;
vary,   name=kqtf, step=1.0E-7 ;
vary,   name=kqtd, step=1.0E-7 ;
lmdif,  calls=500, tolerance=1.0E-21;
endmatch;''')

if sequence_to_track=='lhcb1':
    matching_Q_Qp(mad=mad_track,q_knob_1='dQx.b1',q_knob_2='dQy.b1',qp_knob_1='dQpx.b1',qp_knob_2='dQpy.b1')
else:
    matching_Q_Qp(mad=mad_track,q_knob_1='dQx.b2',q_knob_2='dQy.b2',qp_knob_1='dQpx.b2',qp_knob_2='dQpy.b2')


mad_track.call("modules/submodule_05e_corrvalue.madx")
mad_track.call("modules/submodule_05f_final.madx")


twiss_dfs, other_data = ost.twiss_with_no_use(mad_track, [sequence_to_track],
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_final_with_BB',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=False, check_separations_at_ips=False)

mad_track.globals['on_bb_charge']=0

twiss_dfs, other_data = ost.twiss_with_no_use(mad_track, [sequence_to_track],
        tol_beta=tol_beta, tol_sep=tol_sep,
        twiss_fname='twiss_final_without_BB',
        save_twiss_files=save_intermediate_twiss,
        check_betas_at_ips=False, check_separations_at_ips=False)

mad_track.globals['on_bb_charge']=1
pd.DataFrame([mad_track.globals]).to_parquet('final_globals.parquet')

my_variables=mad_track.get_variables_dataframes()

my_variables['constants'].to_parquet('final_constants.parquet')
my_variables['independent_variables'].to_parquet('final_independent_variables.parquet')
my_variables['dependent_variables'].to_parquet('final_dependent_variables.parquet')


pd.DataFrame([python_parameters]).to_pickle('final_python_parameters.pickle')
pd.DataFrame([mask_parameters]).to_pickle('final_mask_parameters.pickle')
pd.DataFrame([knob_parameters]).to_pickle('final_knob_parameters.pickle')
os.chdir(python_parameters['parent_folder'])

# Generate sixtrack
if False:    
    if enable_bb_legacy:
        mad_track.call("modules/module_06_generate.madx")
    else:
        pm.generate_sixtrack_input(mad_track,
            seq_name=sequence_to_track,
            bb_df=bb_df_track,
            output_folder='./',
            reference_bunch_charge_sixtrack_ppb=(
                mad_track.sequence[sequence_to_track].beam.npart),
            emitnx_sixtrack_um=(
                mad_track.sequence[sequence_to_track].beam.exn),
            emitny_sixtrack_um=(
                mad_track.sequence[sequence_to_track].beam.eyn),
            sigz_sixtrack_m=(
                mad_track.sequence[sequence_to_track].beam.sigt),
            sige_sixtrack=(
                mad_track.sequence[sequence_to_track].beam.sige),
            ibeco_sixtrack=1,
            ibtyp_sixtrack=0,
            lhc_sixtrack=2,
            ibbc_sixtrack=0,
            radius_sixtrack_multip_conversion_mad=0.017,
            skip_mad_use=True)

    # Get optics and orbit at start ring
    optics_orbit_start_ring = pm.get_optics_and_orbit_at_start_ring(
        mad_track, sequence_to_track, skip_mad_use=True)
    with open('./optics_orbit_at_start_ring.pkl', 'wb') as fid:
        pickle.dump(optics_orbit_start_ring, fid)

    # Generate pysixtrack lines
    if enable_bb_legacy:
        print('Pysixtrack line is not generated with bb legacy macros')
    else:
        pysix_fol_name = "./pysixtrack"
        dct_pysxt = pm.generate_pysixtrack_line_with_bb(mad_track,
            sequence_to_track, bb_df_track,
            closed_orbit_method='from_mad',
            pickle_lines_in_folder=pysix_fol_name,
            skip_mad_use=True)
