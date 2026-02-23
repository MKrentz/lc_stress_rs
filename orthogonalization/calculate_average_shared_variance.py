import numpy as np
import nibabel as nib
import glob
from nilearn import plotting
import matplotlib.pyplot as plt

# Specify the base path for your subjects
base_path = '/project/3013068.03/resting_state'
subject_list = [i[-7:] for i in glob.glob('/project/3013068.03/resting_state/sub*')]
exclusion_list = ['sub-006', 'sub-008', 'sub-010']
subject_list = [i for i in subject_list if i not in exclusion_list]

difference_files = []

# Collect all the paths for the difference files
for sub_id in subject_list:
    difference_file_path = f'{base_path}/{sub_id}/difference.nii.gz'
    difference_files.append(difference_file_path)

# Load the difference images and calculate the mean
diff_images = [nib.load(file).get_fdata() for file in difference_files]

# Average the images
average_image = np.mean(diff_images, axis=0)

# Create a NIfTI image from the average data
average_nifti_image = nib.Nifti1Image(average_image, affine=nib.load(difference_files[0]).affine)

# Save the average image
average_nifti_path = f'{base_path}/average_difference.nii.gz'
nib.save(average_nifti_image, average_nifti_path)

print(f"Average difference image saved as: {average_nifti_path}")

# Plot the average image on a glass brain
plotting.plot_glass_brain(average_nifti_path, title="Average Difference Map", colorbar=True, threshold=1.0)
plt.show()  # Display the plot
plt.savefig(f'{base_path}/average_difference.png')