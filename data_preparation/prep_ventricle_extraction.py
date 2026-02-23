import glob
import nibabel as nib
import nilearn.image
from nilearn import maskers
from Subject_Class_new import Subject
import numpy as np
import pandas as pd

basepath = '/project/3013068.03/resting_state/'
part_list = glob.glob(basepath+ 'sub-*')
part_list.sort()
part_list = [i[-7:] for i in part_list]
run_list = [1, 2]
session_list = [1, 2]

for part in part_list:
    sub = Subject(part)
    vent_mask = nib.load(f'/project/3013068.03/stats/4thVentricleMasks/resampled_shrunk_masks/resampled_4thVentricleMask_{part}.nii.gz')
    vent_masker = nilearn.maskers.NiftiMasker(mask_img=vent_mask, t_r=2.02, verbose=1, standardize=True)
    for session in session_list:
        for run in run_list:
            sub_data = sub.get_func_data(AROMA=False, MNI=False, task='RS', session=session, run=run)
            sub_vent_data = vent_masker.fit_transform(sub_data)
            sub_vent_data_mean = pd.DataFrame(np.mean(sub_vent_data, axis=1), columns= ['z_mean_vent'])
            sub_vent_data_mean.to_csv(f'/project/3013068.03/resting_state/ventricle_timecourses/vent_zmean_timecourse_native_'
                                      f'{part}_session-{session}_run-{run}.csv')
