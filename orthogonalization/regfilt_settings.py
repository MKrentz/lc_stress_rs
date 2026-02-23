import pandas as pd
import glob
import os
import subprocess
import sys

sub_id = sys.argv[1]
run_list = [1, 2]
session_list = [1, 2]
space_list = ['mni', 'native']
subprocess.call('module load fsl', shell=True)
for session in session_list:
    for run in run_list:
        for space in space_list:
            filter_list = pd.read_csv(glob.glob(f'/project/3013068.03/fmriprep_test/{sub_id}/ses-mri0{session+1}/func/'
                                            f'{sub_id}_ses-mri0{session+1}_*28RS_*_'
                                            f'run-{run}_AROMAnoiseICs.csv')[0], header=None).iloc[0]
            filter_list = filter_list.tolist()
            filter_list = ", ".join(map(str, filter_list))
            input_path = glob.glob(f'/project/3013068.03/resting_state/{sub_id}/smoothed_imgs/'
                                   f'func_data_{space}_*_{sub_id}_session-{session}_run-{run}.nii.gz')[0]
            design_input = f'/project/3013068.03/resting_state/{sub_id}/ortho_mm/orthogonalized_mm_run-{run}_session{session}.tsv'
            output_path = '/project/3013068.03/resting_state/'

            command = ' '.join(['fsl_regfilt',
                      '--in=' + input_path,
                      '--design=' + design_input,
                      '--filter="' + filter_list + '"',
                      '--out=' + os.path.join(output_path, sub_id, f'aroma_cleaned_data/denoised_func_data_{space}_nonaggr_'
                                                           f'retroortho_session-{session}_run-{run}.nii.gz')])

            subprocess.call(command, shell=True)