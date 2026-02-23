import pandas as pd
from nilearn.glm import first_level
from nilearn.image import load_img
import nibabel as nib
import numpy as np
import glob
import sys

sub_id = sys.argv[1]

# Paths to your data
before_ortho_path = glob.glob(f'/project/3013068.03/fmriprep_test/{sub_id}/ses-mri02/func/'
                     f'{sub_id}_ses-mri02_task-*28RS*'
                     f'_run-1_desc-MELODIC_mixing.tsv')[0]
after_ortho_path = f"/project/3013068.03/resting_state/{sub_id}/ortho_mm/orthogonalized_mm_run-1_session1.csv"
fmri_image_path = glob.glob(f'/project/3013068.03/fmriprep_test/{sub_id}/ses-mri02/func/'
                   f'{sub_id}_ses-mri02_*28RS*_run-1*MNI*_desc-preproc_bold.nii.gz')[0]  # Replace with your fMRI image path

# Load mixing matrices
mm_before_df = pd.read_csv(before_ortho_path, header=None, delim_whitespace=True)
mm_after_df = pd.read_csv(after_ortho_path, header=0, index_col=0)

# Load noise components
noise_df = pd.read_csv(glob.glob(f'/project/3013068.03/fmriprep_test/{sub_id}/ses-mri02/func/'
                                 f'{sub_id}_ses-mri02_*E28RS*run-1_AROMAnoiseICs.csv')[0],
                       header=None)

# Extract the first row of the noise DataFrame as a list and adjust indexing
noise_list = noise_df.iloc[0].tolist()  # Convert first row to a list
noise_list = [i - 1 for i in noise_list]  # Adjust from 1-based to 0-based indexing

t_r = 2.02  # repetition time is 1 second
n_scans = 240  # the acquisition comprises 128 scans
frame_times = (
    np.arange(n_scans) * t_r
)

# Adding intercept terms
design_matrix_before = mm_before_df.copy().iloc[:, noise_list]  # Select columns from noise_list
design_matrix_before['constant'] = 1
design_matrix_before.index = frame_times

design_matrix_after = mm_after_df.copy().iloc[:, noise_list]  # Select noise components only from after orthogonalization
design_matrix_after['constant'] = 1
design_matrix_after.index = frame_times

# Load the fMRI image
fmri_img = load_img(fmri_image_path)

# Fit the GLM using the noise components
model_before = first_level.FirstLevelModel(smoothing_fwhm=6)
model_after = first_level.FirstLevelModel(smoothing_fwhm=6)

# Fit both models
model_before = model_before.fit(fmri_img, design_matrices=design_matrix_before)
model_after = model_after.fit(fmri_img, design_matrices=design_matrix_after)

# Define contrasts for all noise predictors (the last column must be Signal predictors)
# Assuming you want to compare the variance explained by noise predictors only
contrast_vector = [1] * design_matrix_before.shape[1]  # This assumes all noise predictors are of interest
contrast_vector[-1] = 0 
# Compute the contrasts for both models
contrast_map_before = model_before.compute_contrast([contrast_vector], output_type='z_score')
contrast_map_after = model_after.compute_contrast([contrast_vector], output_type='z_score')

# Calculate the difference in variance explained
contrast_diff = contrast_map_before.get_fdata() - contrast_map_after.get_fdata()

nib.save(contrast_map_before, f'/project/3013068.03/resting_state/{sub_id}/before.nii.gz')
nib.save(contrast_map_after, f'/project/3013068.03/resting_state/{sub_id}/after.nii.gz')
nib.save(nib.Nifti1Image(contrast_diff, affine=contrast_map_before.affine), f'/project/3013068.03/resting_state/{sub_id}/difference.nii.gz')
