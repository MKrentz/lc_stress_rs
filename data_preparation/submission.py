import glob
import subprocess

scriptname = 'rs_confound_creation'
memory_req = '24000mb'
walltime = '02:15:00'
#walltime = '00:25:00'

subject_list = [i[-7:] for i in glob.glob('/project/3013068.03/resting_state/sub*')]
exclusion_list = ['sub-006', 'sub-008', 'sub-010']
subject_list = [i for i in subject_list if i not in exclusion_list]
#subject_list = subject_list[:2]
#subject_list = ['sub-001', 'sub-002', 'sub-003']
environment_string = '(echo "source activate ./.conda/envs/venv2/"; echo sleep 5; '
for sub_id in subject_list:
    script_string = f'echo python /home/path/to/repository/github_rep/stress_resting_state/{scriptname}.py {sub_id})'
    submit_command = f" | qsub -N {sub_id} -l 'nodes=1, mem={memory_req}, walltime={walltime}'"
    subprocess.call(environment_string + script_string + submit_command, shell=True)
