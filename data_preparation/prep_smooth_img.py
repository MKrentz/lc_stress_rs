from nilearn.image import smooth_img
import nibabel as nib
from Subject_Class_new import Subject
import sys

sub_id = sys.argv[1]
session_list = [1, 2]
run_list = [1, 2]
sub = Subject(sub_id)

for session in session_list:
    for run in run_list:
        sub_mni_data_smoothed = smooth_img(sub.get_func_data(MNI=True, AROMA=False, run=run, session=session, task='RS'),
                                          fwhm=6)
        sub_native_data_smoothed = smooth_img(sub.get_func_data(MNI=False, AROMA=False, run=run, session=session, task='RS'),
                                          fwhm=3)
        nib.save(sub_mni_data_smoothed, f'/project/3013068.03/resting_state/{sub_id}/smoothed_imgs/'
                                        f'func_data_mni_6mm_{sub_id}_session-{session}_run-{run}.nii.gz')
        nib.save(sub_native_data_smoothed, f'/project/3013068.03/resting_state/{sub_id}/smoothed_imgs/'
                                        f'func_data_native_3mm_{sub_id}_session-{session}_run-{run}.nii.gz')
