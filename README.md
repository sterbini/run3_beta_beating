# run3_beta_beating
Simulation setup for Run 3 beta beating.


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

### Postprocessing
You can do some postprocessing on SWAN.
