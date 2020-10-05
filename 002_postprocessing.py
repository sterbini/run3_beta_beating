import pandas as pd

jobs_df=pd.read_pickle('data/input_jobs_df.pickle')
b1_df=jobs_df[jobs_df['mode']=='b1_with_bb'].copy()
b2_df=jobs_df[jobs_df['mode']=='b4_from_b2_with_bb'].copy()

def get_variables(df,variable='python'):
    my_list=[]
    for ii in df.working_folder.values:
        my_list.append(pd.read_pickle(ii+'/final_' + variable + '_parameters.pickle'))
    return pd.concat(my_list)
	

def get_final_summary(df, bb=False):
    my_list=[]

    for ii in df.working_folder.values:
        if ii[0:2]=='b1':
            aux='b1'
        else:
            aux='b2'
        if not bb:
            my_list.append(pd.read_parquet(ii+'/twiss_final_without_BB_summary_lhc'+ aux  +'.parquet'))
        else:
            my_list.append(pd.read_parquet(ii+'/twiss_final_with_BB_summary_lhc'+ aux  +'.parquet'))

    return pd.concat(my_list)

def get_final_twiss(df, bb=False):
    my_list=[]

    for ii in df.working_folder.values:
        if ii[0:2]=='b1':
            aux='b1'
        else:
            aux='b2'
        if not bb:
            my_list.append(pd.read_parquet(ii+'/twiss_final_without_BB_seq_lhc'+ aux  +'.parquet'))
        else:
            my_list.append(pd.read_parquet(ii+'/twiss_final_with_BB_seq_lhc'+ aux  +'.parquet'))

    return pd.concat(my_list)
#pd.read_parquet('./twiss_final_with_BB_summary_lhcb1.parquet').columns
