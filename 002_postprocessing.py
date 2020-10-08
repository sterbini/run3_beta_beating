import pandas as pd
import bokeh

jobs_df=pd.read_pickle('/eos/user/g/gsterb/run3_beta_beating/2020.10.08.09.28.48_b1_optics_20_mixed_2.5/input_jobs_df.pickle')
b1_df=jobs_df[jobs_df['mode']=='b1_with_bb'].copy()
b2_df=jobs_df[jobs_df['mode']=='b4_from_b2_with_bb'].copy()

def get_variables(df,variable='python'):
    my_list=[]
    for ii in df.working_folder.values:
        my_list.append(pd.read_pickle(ii+'/final_' + variable + '_parameters.pickle'))
    return pd.concat(my_list)
	

def get_final_summary(df, bb=False):
    my_list=[]

    for ii, jj, zz in zip(df['working_folder'], df['beta_ref'], df['mode']):
        if zz[0:2]=='b1':
            aux='b1'
        else:
            aux='b2'
        if not bb:
            my_df=pd.read_parquet(ii+'/twiss_final_without_BB_summary_lhc'+ aux  +'.parquet')
            my_df['beta_ref']=jj		
            my_list.append(my_df)
        else:
            my_df=pd.read_parquet(ii+'/twiss_final_with_BB_summary_lhc'+ aux  +'.parquet')
            my_df['beta_ref']=jj		
            my_list.append(my_df)

    return pd.concat(my_list)

def get_final_twiss(df, bb=False):
    my_list=[]

    for ii, jj, zz in zip(df['working_folder'], df['beta_ref'], df['mode']):
        if zz[0:2]=='b1':
            aux='b1'
        else:
            aux='b2'
        if not bb:
            my_df=pd.read_parquet(ii+'/twiss_final_without_BB_seq_lhc'+ aux  +'.parquet')
            my_df['beta_ref']=jj		
            my_list.append(my_df)
        else:
            my_df=pd.read_parquet(ii+'/twiss_final_with_BB_seq_lhc'+ aux  +'.parquet')
            my_df['beta_ref']=jj		
            my_list.append(my_df)

    return pd.concat(my_list)

### 
beam='b1'


if beam=='b1':
    df=b1_df
else:
    df=b2_df
    
py_df=get_variables(df)
mask_df=get_variables(df,variable='mask')
summary_df_wo_bb = get_final_summary(py_df, bb = False)
summary_df_w_bb  = get_final_summary(py_df, bb = True)
final_twiss_wo_bb = get_final_twiss(py_df, bb = False)
final_twiss_w_bb  = get_final_twiss(py_df, bb = True)
py_df['par_beam_npart']=mask_df['par_beam_npart']
py_df['par_beam_sigt']=mask_df['par_beam_sigt']
final_twiss_w_bb['beta_beating_x']=((final_twiss_w_bb.betx-final_twiss_wo_bb.betx)/final_twiss_wo_bb.betx)
final_twiss_w_bb['beta_beating_y']=((final_twiss_w_bb.bety-final_twiss_wo_bb.bety)/final_twiss_wo_bb.bety)


import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save
from bokeh.models import  Label, LabelSet, DataRange1d, ColumnDataSource, DataTable, DateFormatter, TableColumn
from bokeh.layouts import column

# Luminosity
TOOLTIPS = [
    ("beta_ref", "@beta_ref [m]"),
    ("N", "@N [1e11 ppb]"),
    ("sigma_z", "@sigma_z [cm]"),
    ("L at IP1", "@LIP1{1.111111} [1e34 Hz/cm^2]"),
    ("L at IP2", "@LIP2{1.111111} [1e34 Hz/cm^2]"),
    ("L at IP5", "@LIP5{1.111111} [1e34 Hz/cm^2]"),
    ("L at IP8", "@LIP8{1.111111} [1e34 Hz/cm^2]"),
    ("file", "@optics_file"),
]
output_file(f"/eos/user/g/gsterb/run3_beta_beating_plots/{beam}.html")

source = ColumnDataSource(data=dict(
    beta_ref=py_df['beta_ref'],
    LIP1=py_df['L_IP1']/1e34,
    LIP5=py_df['L_IP5']/1e34,
    LIP2=py_df['L_IP2']/1e34,
    LIP8=py_df['L_IP8']/1e34,
    N=py_df['par_beam_npart']/1e11,
    sigma_z=py_df['par_beam_sigt']*100,
    optics_file=py_df['optics_file'],
    q1_wo_bb=summary_df_wo_bb['q1'],
    q2_wo_bb=summary_df_wo_bb['q2'],
    q1_w_bb=summary_df_w_bb['q1'],
    q2_w_bb=summary_df_w_bb['q2'],
    dq1_wo_bb=summary_df_wo_bb['dq1'],
    dq2_wo_bb=summary_df_wo_bb['dq2'],
    dq1_w_bb=summary_df_w_bb['dq1'],
    dq2_w_bb=summary_df_w_bb['dq2'],
    ip1_px_wo_bb=final_twiss_wo_bb.loc['ip1:1'].px*1e6,
    ip1_px_w_bb=final_twiss_w_bb.loc['ip1:1'].px*1e6,
    ip1_py_wo_bb=final_twiss_wo_bb.loc['ip1:1'].py*1e6,
    ip1_py_w_bb=final_twiss_w_bb.loc['ip1:1'].py*1e6,
    ip2_px_wo_bb=final_twiss_wo_bb.loc['ip2:1'].px*1e6,
    ip2_px_w_bb=final_twiss_w_bb.loc['ip2:1'].px*1e6,
    ip2_py_wo_bb=final_twiss_wo_bb.loc['ip2:1'].py*1e6,
    ip2_py_w_bb=final_twiss_w_bb.loc['ip2:1'].py*1e6,
    ip5_px_wo_bb=final_twiss_wo_bb.loc['ip5:1'].px*1e6,
    ip5_px_w_bb=final_twiss_w_bb.loc['ip5:1'].px*1e6,
    ip5_py_wo_bb=final_twiss_wo_bb.loc['ip5:1'].py*1e6,
    ip5_py_w_bb=final_twiss_w_bb.loc['ip5:1'].py*1e6,
    ip8_px_wo_bb=final_twiss_wo_bb.loc['ip8:1'].px*1e6,
    ip8_px_w_bb=final_twiss_w_bb.loc['ip8:1'].px*1e6,
    ip8_py_wo_bb=final_twiss_wo_bb.loc['ip8:1'].py*1e6,
    ip8_py_w_bb=final_twiss_w_bb.loc['ip8:1'].py*1e6,
    ip1_x_wo_bb=final_twiss_wo_bb.loc['ip1:1'].x*1e6,
    ip1_x_w_bb=final_twiss_w_bb.loc['ip1:1'].x*1e6,
    ip1_y_wo_bb=final_twiss_wo_bb.loc['ip1:1'].y*1e6,
    ip1_y_w_bb=final_twiss_w_bb.loc['ip1:1'].y*1e6,
    ip2_x_wo_bb=final_twiss_wo_bb.loc['ip2:1'].x*1e6,
    ip2_x_w_bb=final_twiss_w_bb.loc['ip2:1'].x*1e6,
    ip2_y_wo_bb=final_twiss_wo_bb.loc['ip2:1'].y*1e6,
    ip2_y_w_bb=final_twiss_w_bb.loc['ip2:1'].y*1e6,
    ip5_x_wo_bb=final_twiss_wo_bb.loc['ip5:1'].x*1e6,
    ip5_x_w_bb=final_twiss_w_bb.loc['ip5:1'].x*1e6,
    ip5_y_wo_bb=final_twiss_wo_bb.loc['ip5:1'].y*1e6,
    ip5_y_w_bb=final_twiss_w_bb.loc['ip5:1'].y*1e6,
    ip8_x_wo_bb=final_twiss_wo_bb.loc['ip8:1'].x*1e6,
    ip8_x_w_bb=final_twiss_w_bb.loc['ip8:1'].x*1e6,
    ip8_y_wo_bb=final_twiss_wo_bb.loc['ip8:1'].y*1e6,
    ip8_y_w_bb=final_twiss_w_bb.loc['ip8:1'].y*1e6,
    phase_ip15_x_wo_bb=final_twiss_wo_bb.loc['ip1:1'].mux.values-final_twiss_wo_bb.loc['ip5:1'].mux.values,
    phase_ip15_x_w_bb=final_twiss_w_bb.loc['ip1:1'].mux.values-final_twiss_w_bb.loc['ip5:1'].mux.values,
    phase_ip15_y_wo_bb=final_twiss_wo_bb.loc['ip1:1'].muy.values-final_twiss_wo_bb.loc['ip5:1'].muy.values,
    phase_ip15_y_w_bb=final_twiss_w_bb.loc['ip1:1'].muy.values-final_twiss_w_bb.loc['ip5:1'].muy.values,
    max_betx_beating=final_twiss_w_bb.groupby('beta_ref')['beta_beating_x'].max().values[::-1],
    max_bety_beating=final_twiss_w_bb.groupby('beta_ref')['beta_beating_y'].max().values[::-1],
))


########
my_plot_L = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'B1/B2 luminosities',x_axis_label='reference beta [m]', y_axis_label='[1e34 Hz/cm^2]', y_range=DataRange1d(only_visible=True))
my_plot_L.line('beta_ref', 'LIP1',  line_width=2, source=source, color="blue")
my_plot_L.line('beta_ref', 'LIP5',  line_width=2, source=source, color="red")
my_plot_L.circle('beta_ref', 'LIP1',  line_width=2, source=source, legend_label='Luminosity in IP1', color="blue")
my_plot_L.circle('beta_ref', 'LIP5',  line_width=2, source=source, legend_label='Luminosity in IP5', color="red")
my_plot_L.legend.location = "top_left"
my_plot_L.legend.click_policy="hide"

########
my_plot_N = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} bunch population',x_axis_label='reference beta [m]', y_axis_label='[1e11 ppb]', y_range=DataRange1d(only_visible=True))
my_plot_N.line('beta_ref', 'N',  line_width=2, source=source, color="black")
my_plot_N.circle('beta_ref', 'N',  line_width=2, source=source, color="black")

########
my_plot_sigma = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} sigma_z', x_axis_label='reference beta [m]', y_axis_label='[cm]', y_range=DataRange1d(only_visible=True))
my_plot_sigma.line('beta_ref', 'sigma_z',  line_width=2, source=source, color="black")
my_plot_sigma.circle('beta_ref', 'sigma_z',  line_width=2, source=source, color="black")

#######
# q
#######

my_plot_q = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} tune', x_axis_label='reference beta [m]', y_axis_label='', y_range=DataRange1d(only_visible=True))
my_plot_q.circle('beta_ref', 'q1_wo_bb',  line_width=2, source=source, legend_label='Q1 w/o BB', color="blue")
my_plot_q.circle('beta_ref', 'q2_wo_bb',  line_width=2, source=source, legend_label='Q2 w/o BB', color="red")
my_plot_q.square('beta_ref', 'q1_w_bb',  line_width=2, source=source, legend_label='Q1 w/ BB', color="blue")
my_plot_q.square('beta_ref', 'q2_w_bb',  line_width=2, source=source, legend_label='Q2 w/ BB', color="red")
my_plot_q.legend.location = "top_left"
my_plot_q.legend.click_policy="hide"

#######
# dq
#######

my_plot_dq = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} chromaticity', x_axis_label='reference beta [m]', y_axis_label='', y_range=DataRange1d(only_visible=True))
my_plot_dq.circle('beta_ref', 'dq1_wo_bb',  line_width=2, source=source, legend_label='dQ1 w/o BB', color="blue")
my_plot_dq.circle('beta_ref', 'dq2_wo_bb',  line_width=2, source=source, legend_label='dQ2 w/o BB', color="red")
my_plot_dq.square('beta_ref', 'dq1_w_bb',  line_width=2, source=source, legend_label='dQ1 w/ BB', color="blue")
my_plot_dq.square('beta_ref', 'dq2_w_bb',  line_width=2, source=source, legend_label='dQ2 w/ BB', color="red")
my_plot_dq.legend.location = "top_left"
my_plot_dq.legend.click_policy="hide"


#######
# xing IP1
#######

my_plot_x_ip1 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP1 crossing angle', x_axis_label='reference beta [m]', y_axis_label='[urad]', y_range=DataRange1d(only_visible=True))
my_plot_x_ip1.circle('beta_ref', 'ip1_px_wo_bb',  line_width=2, source=source, legend_label='px w/o BB', color="blue")
my_plot_x_ip1.circle('beta_ref', 'ip1_py_wo_bb',  line_width=2, source=source, legend_label='py w/o BB', color="red")
my_plot_x_ip1.square('beta_ref', 'ip1_px_w_bb',  line_width=2, source=source, legend_label='px w/ BB', color="blue")
my_plot_x_ip1.square('beta_ref', 'ip1_py_w_bb',  line_width=2, source=source, legend_label='py w/ BB', color="red")
my_plot_x_ip1.legend.location = "top_left"
my_plot_x_ip1.legend.click_policy="hide"

#######
# offset IP1
#######

my_plot_xo_ip1 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP1 offset', x_axis_label='reference beta [m]', y_axis_label='[um]', y_range=DataRange1d(only_visible=True))
my_plot_xo_ip1.circle('beta_ref', 'ip1_x_wo_bb',  line_width=2, source=source, legend_label='x w/o BB', color="blue")
my_plot_xo_ip1.circle('beta_ref', 'ip1_y_wo_bb',  line_width=2, source=source, legend_label='y w/o BB', color="red")
my_plot_xo_ip1.square('beta_ref', 'ip1_x_w_bb',  line_width=2, source=source, legend_label='x w/ BB', color="blue")
my_plot_xo_ip1.square('beta_ref', 'ip1_y_w_bb',  line_width=2, source=source, legend_label='y w/ BB', color="red")
my_plot_xo_ip1.legend.location = "top_left"
my_plot_xo_ip1.legend.click_policy="hide"

#######
# xing IP2
#######

my_plot_x_ip2 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP2 crossing angle', x_axis_label='reference beta [m]', y_axis_label='[urad]', y_range=DataRange1d(only_visible=True))
my_plot_x_ip2.circle('beta_ref', 'ip2_px_wo_bb',  line_width=2, source=source, legend_label='px w/o BB', color="blue")
my_plot_x_ip2.circle('beta_ref', 'ip2_py_wo_bb',  line_width=2, source=source, legend_label='py w/o BB', color="red")
my_plot_x_ip2.square('beta_ref', 'ip2_px_w_bb',  line_width=2, source=source, legend_label='px w/ BB', color="blue")
my_plot_x_ip2.square('beta_ref', 'ip2_py_w_bb',  line_width=2, source=source, legend_label='py w/ BB', color="red")
my_plot_x_ip2.legend.location = "top_left"
my_plot_x_ip2.legend.click_policy="hide"

#######
# offset IP2
#######

my_plot_xo_ip2 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP2 offset', x_axis_label='reference beta [m]', y_axis_label='[um]', y_range=DataRange1d(only_visible=True))
my_plot_xo_ip2.circle('beta_ref', 'ip2_x_wo_bb',  line_width=2, source=source, legend_label='x w/o BB', color="blue")
my_plot_xo_ip2.circle('beta_ref', 'ip2_y_wo_bb',  line_width=2, source=source, legend_label='y w/o BB', color="red")
my_plot_xo_ip2.square('beta_ref', 'ip2_x_w_bb',  line_width=2, source=source, legend_label='x w/ BB', color="blue")
my_plot_xo_ip2.square('beta_ref', 'ip2_y_w_bb',  line_width=2, source=source, legend_label='y w/ BB', color="red")
my_plot_xo_ip2.legend.location = "top_left"
my_plot_xo_ip2.legend.click_policy="hide"

#######
# xing IP5
#######

my_plot_x_ip5 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP5 crossing angle', x_axis_label='reference beta [m]', y_axis_label='[urad]', y_range=DataRange1d(only_visible=True))
my_plot_x_ip5.circle('beta_ref', 'ip5_px_wo_bb',  line_width=2, source=source, legend_label='px w/o BB', color="blue")
my_plot_x_ip5.circle('beta_ref', 'ip5_py_wo_bb',  line_width=2, source=source, legend_label='py w/o BB', color="red")
my_plot_x_ip5.square('beta_ref', 'ip5_px_w_bb',  line_width=2, source=source, legend_label='px w/ BB', color="blue")
my_plot_x_ip5.square('beta_ref', 'ip5_py_w_bb',  line_width=2, source=source, legend_label='py w/ BB', color="red")
my_plot_x_ip5.legend.location = "top_left"
my_plot_x_ip5.legend.click_policy="hide"


#######
# offset IP5
#######

my_plot_xo_ip5 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP5 offset', x_axis_label='reference beta [m]', y_axis_label='[um]', y_range=DataRange1d(only_visible=True))
my_plot_xo_ip5.circle('beta_ref', 'ip5_x_wo_bb',  line_width=2, source=source, legend_label='x w/o BB', color="blue")
my_plot_xo_ip5.circle('beta_ref', 'ip5_y_wo_bb',  line_width=2, source=source, legend_label='y w/o BB', color="red")
my_plot_xo_ip5.square('beta_ref', 'ip5_x_w_bb',  line_width=2, source=source, legend_label='x w/ BB', color="blue")
my_plot_xo_ip5.square('beta_ref', 'ip5_y_w_bb',  line_width=2, source=source, legend_label='y w/ BB', color="red")
my_plot_xo_ip5.legend.location = "top_left"
my_plot_xo_ip5.legend.click_policy="hide"

#######
# xing IP8
#######

my_plot_x_ip8 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP8 crossing angle', x_axis_label='reference beta [m]', y_axis_label='[urad]', y_range=DataRange1d(only_visible=True))
my_plot_x_ip8.circle('beta_ref', 'ip8_px_wo_bb',  line_width=2, source=source, legend_label='px w/o BB', color="blue")
my_plot_x_ip8.circle('beta_ref', 'ip8_py_wo_bb',  line_width=2, source=source, legend_label='py w/o BB', color="red")
my_plot_x_ip8.square('beta_ref', 'ip8_px_w_bb',  line_width=2, source=source, legend_label='px w/ BB', color="blue")
my_plot_x_ip8.square('beta_ref', 'ip8_py_w_bb',  line_width=2, source=source, legend_label='py w/ BB', color="red")
my_plot_x_ip8.legend.location = "top_left"
my_plot_x_ip8.legend.click_policy="hide"

#######
# offset IP8
#######

my_plot_xo_ip8 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} IP8 offset', x_axis_label='reference beta [m]', y_axis_label='[um]', y_range=DataRange1d(only_visible=True))
my_plot_xo_ip8.circle('beta_ref', 'ip8_x_wo_bb',  line_width=2, source=source, legend_label='x w/o BB', color="blue")
my_plot_xo_ip8.circle('beta_ref', 'ip8_y_wo_bb',  line_width=2, source=source, legend_label='y w/o BB', color="red")
my_plot_xo_ip8.square('beta_ref', 'ip8_x_w_bb',  line_width=2, source=source, legend_label='x w/ BB', color="blue")
my_plot_xo_ip8.square('beta_ref', 'ip8_y_w_bb',  line_width=2, source=source, legend_label='y w/ BB', color="red")
my_plot_xo_ip8.legend.location = "top_left"
my_plot_xo_ip8.legend.click_policy="hide"

#######
# phase IP1->IP5
#######

my_plot_phase_ip15 = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} Delta mu IP1-IP5', x_axis_label='reference beta [m]', y_axis_label='[2 pi]', y_range=DataRange1d(only_visible=True))
my_plot_phase_ip15.circle('beta_ref', 'phase_ip15_x_wo_bb',  line_width=2, source=source, legend_label='delta mux w/o BB', color="blue")
my_plot_phase_ip15.circle('beta_ref', 'phase_ip15_y_wo_bb',  line_width=2, source=source, legend_label='delta muy w/o BB', color="red")
my_plot_phase_ip15.square('beta_ref', 'phase_ip15_x_w_bb',  line_width=2, source=source, legend_label='delta mux w/ BB', color="blue")
my_plot_phase_ip15.square('beta_ref', 'phase_ip15_y_w_bb',  line_width=2, source=source, legend_label='delta muy w/ BB', color="red")
my_plot_phase_ip15.legend.location = "top_left"
my_plot_phase_ip15.legend.click_policy="hide"

#######
# beta beating
#######

my_plot_beating = figure(plot_width=1200, plot_height=300, tooltips=TOOLTIPS,
           title=f'{beam} maximum beta beating', x_axis_label='reference beta [m]', y_axis_label='', y_range=DataRange1d(only_visible=True))
my_plot_beating.circle('beta_ref', 'max_betx_beating',  line_width=2, source=source, legend_label='x-beating', color="blue")
my_plot_beating.circle('beta_ref', 'max_bety_beating',  line_width=2, source=source, legend_label='y-beating', color="red")
my_plot_beating.legend.location = "top_left"
my_plot_beating.legend.click_policy="hide"


save(column(my_plot_N, my_plot_sigma, my_plot_L, my_plot_q, my_plot_dq,
    my_plot_x_ip1, my_plot_xo_ip1, my_plot_x_ip2,my_plot_xo_ip2, my_plot_x_ip5,my_plot_xo_ip5, my_plot_x_ip8, my_plot_xo_ip8,
    my_plot_phase_ip15, my_plot_beating))
