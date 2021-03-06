from pyHTC.Study import *
import pyHTC.toolkit as toolkit
import os
import pandas as pd
import sys
from datetime import datetime

now = datetime.now()


date_string = now.strftime("%Y.%m.%d.%H.%M.%S")

my_list=[]
logname=os.getenv("LOGNAME")
from itertools import product

for optics, mode, emittance in product(range(20,31), ['b1_with_bb','b4_from_b2_with_bb'], [1.8]):
    my_list.append({
        'parent_folder': os.getcwd(),
	'working_folder': f'/eos/user/{logname[0]}/{logname}/run3_beta_beating/{date_string}_{mode[0:2]}_optics_{optics}_mixed_{emittance}',
        'mode' : mode,
        'optics_file' : f'opticsfile.{optics}',
	'emittance_um': emittance,
	'date_string':date_string,
        })


pd.DataFrame(my_list).to_pickle(f'./data/input_jobs_df.pickle')


try:
    os.symlink(f'/eos/user/{logname[0]}/{logname}/run3_beta_beating', 'link_to_eos')
except:
    print('EOS link exists.')

mypath=os.getcwd()
python_script=mypath+'/000_job.py'
python_dataframe=mypath+'/data/input_jobs_df.pickle'
python_distribution='/afs/cern.ch/eng/tracking-tools/python_installations/activate_default_python'


myStudy = StudyObj(name='test',
		   path=mypath,
                   job_flavour='microcentury',
                   python_script=python_script,
                   python_distribution=python_distribution,
                   python_dataframe=python_dataframe,
                   arguments='$(ProcId)', queue=len(pd.read_pickle(python_dataframe)), request_cpus=2)

myStudy.describe()

# Preparation of the working directory
toolkit.prepare_work_dir()
# Preparation of the submit file
myStudy.submit2file(myStudy.submit2str())
# Preparation of the bash script
myStudy.script2file(myStudy.script2str())

# Before submitting you should do a minimal check with
# chmod +x myScript.sh
# ./ myScript.sh 0
myStudy.submit2HTCondor()
