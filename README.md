# run3_beta_beating
Simulation setup for Run 3 beta beating.

Here we assume that you have a lxplus login, so you can source a pre-installed conda environment in **afs** and you can run jobs in **HTCondor**.

### Activate the distribution


Login on lxplus.cern.ch
```
ssh lxplus.cern.ch
```

clone this repository in one of you AFS folder

```
git clone https://github.com/sterbini/run3_beta_beating.git
cd run3_beta_beating
```

### Single job test

The first check to do is to run a single job (*000_job.py*). This is run on the lxplus node: so it is a bit slower. 

```
source /afs/cern.ch/eng/tracking-tools/python_installations/activate_default_python
python 000_job.py
```

### Multiple jobs

You can run multiple jobs on HTCondor by
```
source /afs/cern.ch/eng/tracking-tools/python_installations/activate_default_python
python 001_jobs.py  
```
In **001_jobs.py** you can see which how the parameters are varied along the different jobs.

### Postprocessing

You can do some postprocessing (TO BE DONE). 


