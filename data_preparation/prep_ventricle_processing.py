import subprocess
import glob
import nibabel as nib
import nilearn.image
from nilearn import image
from Subject_Class_new import Subject

basepath = '/project/3013068.03/resting_state/'


part_list = glob.glob(basepath+ 'sub-*')
part_list.sort()
for x in part_list:
    sub = Subject(x[-7:])
    bashcommand1 = 'fslmaths ' + f"'/project/3013068.03/stats/4thVentricleMasks/segmented_masks_2024/{x[-7:]}_ses-mri01_acq-t1mpragesagiso08_run-1_label-CSF_probseg_mask.nii.gz'" + \
                   ' -s 0.75 ' f"'/project/3013068.03/stats/4thVentricleMasks/smoothed_masks_2024/{x[-7:]}_T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue_mask_smoothed.nii.gz'" +\
                    ' ; fslmaths ' + f"'/project/3013068.03/stats/4thVentricleMasks/smoothed_masks_2024/{x[-7:]}_T1w_space-MNI152NLin2009cAsym_class-CSF_probtissue_mask_smoothed.nii.gz'" +\
                   ' -thr 0.985 -bin ' f"'/project/3013068.03/stats/4thVentricleMasks/shrunk_masks_2024/4thVentricleMask_{x[-7:]}.nii.gz'"
    subprocess.call(bashcommand1, shell=True)

    vent_mask = nib.load(f'/project/3013068.03/stats/4thVentricleMasks/shrunk_masks_2024/4thVentricleMask_{x[-7:]}.nii.gz')
    func_data = sub.get_func_data(MNI=False, task='RS')
    vent_mask_resampled = nilearn.image.resample_to_img(vent_mask, func_data, interpolation='nearest')
    nib.save(vent_mask_resampled,
             f'/project/3013068.03/stats/4thVentricleMasks/resampled_shrunk_masks/resampled_4thVentricleMask_{x[-7:]}.nii.gz')
