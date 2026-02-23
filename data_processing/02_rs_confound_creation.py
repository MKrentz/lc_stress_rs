import numpy as np
import glob
from Subject_Class_new import Subject
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
import matplotlib.pyplot as plt
import sys
import pandas as pd
from sklearn.decomposition import PCA

#print(sys.argv[1])
#sub_id = sys.argv[1]
sub_id = 'sub-002' #for console execution

fmriprep_path = "/project/3013068.03/fmriprep_test/"
basepath = '/project/3013068.03/resting_state/'
part_list = glob.glob(basepath + 'sub-*')
part_list.sort()

# Indicating subject having the 'stress' condition during their FIRST functional session
stress_list = ['sub-002', 'sub-003', 'sub-004', 'sub-007', 'sub-009', 'sub-013', 'sub-015', 'sub-017', 'sub-021',
               'sub-023', 'sub-025', 'sub-027', 'sub-029']

highpass_filter_names = ['cosine00', 'cosine01', 'cosine02', 'cosine03', 'cosine04',
                         'cosine05']
if sub_id in stress_list:
    session_type = ['stress', 'control']
    print(f'{sub_id} is STRESS FIRST & CONTROL SECOND')
else:
    session_type = ['control', 'stress']
    print(f'{sub_id} is CONTROL FIRST & STRESS SECOND')

sub = Subject(sub_id)
LC_mask = sub.get_LC_resampled(type='linear')  # Function to pull the LC mask in a certain format
LC_mask_data = LC_mask.get_fdata()
LC_mask_bin = LC_mask_data.copy()
LC_mask_bin[LC_mask_bin < 0.1] = 0
LC_mask_bin[LC_mask_bin >= 0.1] = 1
weight_mask = LC_mask_data.copy()
LC_weights = weight_mask[weight_mask >= 0.1]
LC_mask_bin_img = nib.Nifti1Image(LC_mask_bin, affine=LC_mask.affine, header=LC_mask.header)
lc_masker = NiftiMasker(mask_img=LC_mask_bin_img, standardize=True, t_r=2.02, verbose=1)

for session_counter, session in enumerate(session_type):
    for run in range(1, 3):

        # File containing all Shirer Mask ROIs
        mask_df = pd.read_csv('/project/3013068.03/software/Core_Network_ROIs/network_df.csv',
                              index_col=0)

        mask_df['mean_timeseries'] = None
        mask_df['first_eigenvariate'] = None

        print(f'Now processing {sub_id} {session} which is session {session_counter + 1} in run {run}.')
        sub_data_native = nib.load(f'/project/3013068.03/resting_state/{sub_id}/'
                                   f'aroma_cleaned_data/denoised_func_data_native_nonaggr_'
                                   f'retroortho_session-{session_counter + 1}_run-{run}.nii.gz')

        print(f'Functional Data Path: /project/3013068.03/resting_state/{sub_id}/'
              f'aroma_cleaned_data/denoised_func_data_native_nonaggr_'
              f'retroortho_session-{session_counter + 1}_run-{run}.nii.gz')

        sub_data_MNI = nib.load(f'/project/3013068.03/resting_state/{sub_id}/'
                                f'aroma_cleaned_data/denoised_func_data_mni_nonaggr_'
                                f'retroortho_session-{session_counter + 1}_run-{run}.nii.gz')

        fmriprep_confounds = sub.get_confounds(session=session_counter + 1, run=run, task='RS')
        fmriprep_confounds = fmriprep_confounds.fillna(0)

        gs_csf_wm_confounds = pd.read_csv(f'/project/3013068.03/resting_state/{sub_id}/confounds/'
                                          f'confounds_session-{session_counter + 1}_run-{run}.csv',
                                          index_col=0)
        # Selection of nuisance regressors from fmriprep-confound file
        fmriprep_confound_selection = fmriprep_confounds[highpass_filter_names]
        retroicor_confound_selection = sub.get_retroicor_confounds(session=session_counter + 1, run=run, task='RS')
        ventricle_confound = pd.read_csv(f'/project/3013068.03/resting_state/ventricle_timecourses/'
                                         f'vent_zmean_timecourse_native_{sub_id}_session-{session_counter + 1}_run-{run}.csv',
                                         index_col=0)
        joint_confounds = pd.concat([gs_csf_wm_confounds,
                                     fmriprep_confound_selection,
                                     retroicor_confound_selection,
                                     ventricle_confound], axis=1)

        print(f'Using Confounds: {joint_confounds.columns}.')
        for mask_counter, mask in mask_df.iterrows():
            mask_img = nib.load(mask['Path'])
            print(f'Mask loaded for {mask["Path"]} which is {mask["Subnetwork"]} in {mask["Network"]}')
            network_masker = NiftiMasker(mask_img=mask_img, standardize=True, t_r=2.02)

            # Extract timeseries and assign directly without wrapping in a list
            print(f"Extracting Timeseries for {mask['Subnetwork']} in {mask['Network']}")
            extracted_timeseries = network_masker.fit_transform(sub_data_MNI, confounds=joint_confounds)

            # Calculate mean timeseries
            mean_timeseries = np.mean(extracted_timeseries, axis=1)  # Calculate mean across voxels
            mask_df.at[mask_counter, 'mean_timeseries'] = mean_timeseries.flatten()  # Assign mean timeseries
            print(f"Mean Timeseries calculated for {mask['Subnetwork']}")

            # Calculate the first eigenvariate using PCA
            pca = PCA(n_components=1)  # We only want the first component
            first_eigenvariate = pca.fit_transform(extracted_timeseries)
            print(f"First Eigenvariate calculatedfor {mask['Subnetwork']}")
            mask_df.at[mask_counter, 'first_eigenvariate'] = first_eigenvariate.flatten()

        # LC timecourse extraction and voxel averaging
        lc_output = lc_masker.fit_transform(sub_data_native, confounds=joint_confounds)
        lc_output_df = pd.DataFrame(lc_output)
        lc_output_mean = pd.DataFrame(np.average(lc_output, axis=1, weights=LC_weights))

        pca = PCA(n_components=1)  # We only want the first component
        lc_output_eigenvariate = pca.fit_transform(lc_output, 'extracted_timeseries')
        mask_df.loc[len(mask_df)] = [None, 'Locus Coeruleus', 'Locus Coeruleus',
                                     lc_output_mean.values.flatten(), lc_output_eigenvariate.flatten()]

        lc_hist = lc_output_mean.hist(bins=20, figsize=(12, 6), alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'/project/3013068.03/resting_state/{sub_id}/LC_data_hist_{session}_run{run}.png')
        plt.close()

        print("LC Output Mean:", lc_output_mean)
        print("LC Eigenvariate:", lc_output_eigenvariate.flatten())
        mask_df.to_pickle(
            f'/project/3013068.03/resting_state/{sub_id}/GS_extraction_output_{sub_id}_{session}_run-{run}.pkl')