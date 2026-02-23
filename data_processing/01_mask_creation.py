import glob
import nibabel as nib
import numpy as np
import pandas as pd
#sort all pre-ordered network ROIs
mask_list = sorted(glob.glob('/project/3013068.03/software/Core_Network_ROIs/*/*/*_resampled.nii.gz'))
temp_template = nib.load(mask_list[0])
voxel_volume = 1.5 * 1.5 * 1.5  # Volume of one voxel in mm^3

name_dict = {'anterior_Salience': {'1': 'Left Middle Frontal Gyrus',
                                 '2': 'Left Anterior Insula',
                                   '3': 'ACC, MPFC, SMA',
                                   '4': 'Right Middle Frontal Gyrus',
                                   '5': 'Right Anterior Insula',
                                   '6': 'Left Lobule VI, Crus 1',
                                   '7': 'Right Lobule VI, Crus 1'},
             'post_Salience': {'1': 'Left Middle Frontal Gyrus',
                                    '2': 'Left Supramarginal Gyrus, Inferior Parietal Gyrus',
                                    '3': 'Left Precuneus',
                                    '4': 'Right Midcingulate Cortex',
                                    '5': 'Right Superior Parietal Gyrus, Precuneus',
                                    '6': 'Right Supramarginal Gyrus, Inferior Parietal Gyrus',
                                    '7': 'Left Thalamus',
                                    '8': 'Lobule VI',
                                    '9': 'Left Posterior Insula, Putamen',
                                    '10': 'Right Thalamus',
                                    '11': 'Lobule VI',
                                    '12': 'Right Posterior Insula'},
             'LECN': {'1': 'Left Middle Frontal Gyrus, Superior Frontal Gyrus',
                          '2': 'Left Inferior Frontal Gyrus, Orbitofrontal Gyrus',
                          '3': 'Left Superior Parietal Gyrus, Inferior Parietal Gyrus, Precuneus, Angular Gyrus',
                          '4': 'Left Inferior Temporal Gyrus, Middle Temporal Gyrus',
                          '5': 'Right Crus 1',
                          '6': 'Left Thalamus'},
             'RECN': {'1': 'Right Middle Frontal Gyrus, Right Superior Frontal Gyrus',
                           '2': 'Right Middle Frontal Gyrus',
                           '3': 'Right Inferior Parietal Gyrus, Supramarginal Gyrus, Angular Gyrus',
                           '4': 'Right Superior Frontal Gyrus',
                           '5': 'Left Crus I, Crus II, Lobule VI',
                           '6': 'Right Caudate'},
             'ventral_DMN': {'1': 'Left Retrosplenial Cortex, Posterior Cingulate Cortex',
                             '2': 'Left Middle Frontal Gyrus',
                             '3': 'Left Parahippocampal Gyrus',
                             '4': 'Left Middle Occipital Gyrus',
                             '5': 'Right Retrosplenial Cortex, Posterior Cingulate Cortex',
                             '6': 'Precuneus',
                             '7': 'Right Superior Frontal Gyrus, Middle Frontal Gyrus',
                             '8': 'Right Parahippocampal Gyrus',
                             '9': 'Right Angular Gyrus, Middle Occipital Gyrus',
                             '10': 'Right Lobule IX'},
             'dorsal_DMN': {'1': 'Medial Prefrontal Cortex, Anterior Cingulate Cortex, Orbitofrontal Cortex',
                            '2': 'Left Angular Gyrus',
                            '3': 'Right Superior Frontal Gyrus',
                            '4': 'Posterior Cingulate Cortex, Precuneus',
                            '5': 'Midcingulate Cortex',
                            '6': 'Right Angular Gyrus',
                            '7': 'Left and Right Thalamus',
                            '8': 'Left Hippocampus',
                            '9': 'Right Hippocampus'}}

data = []

# Flatten the dictionary into (Path, Subnetwork, Network) pairs
for counter, (network, sub_dict) in enumerate(name_dict.items()):
    for key, value in sub_dict.items():
        # Construct the corresponding path
        if int(key) <= 9:
            path = f'/project/3013068.03/software/Core_Network_ROIs/0{str(int(counter+1))}_{network}/0{key}/{key}_resampled.nii.gz'
        else:
            path = f'/project/3013068.03/software/Core_Network_ROIs/0{str(int(counter+1))}_{network}/{key}/{key}_resampled.nii.gz'
        # Append the row to the data list
        data.append([path, value, network])

# Create the DataFrame
df = pd.DataFrame(data, columns=['Path', 'Subnetwork', 'Network'])
df.to_csv('/project/3013068.03/software/Core_Network_ROIs/network_df.csv')

# Assuming temp_template and df are predefined
temp_template_data = np.round(temp_template.get_fdata(), decimals=0)

# Initialize a list to keep track of overlapping masks along with overlap counts
overlapping_masks = []

# We will store the original mask data as we go
old_mask_data_list = []

# Iterate through the masks in the DataFrame
for mask_counter, row in enumerate(df.iterrows()):
    mask = row[1]['Path']
    current_subnetwork = row[1]['Subnetwork']
    current_network = row[1]['Network']

    old_mask = nib.load(mask)
    old_mask_data = np.round(old_mask.get_fdata(), decimals=0)

    # Check for overlaps with all previously added masks without modifying the original data
    for prev_mask_counter, prev_mask_data in enumerate(old_mask_data_list):
        overlap = (prev_mask_data == 1) & (old_mask_data == 1)  # Overlap condition

        if np.any(overlap):
            # Count the number of overlapping voxels
            overlap_count = np.sum(overlap)
            # Retrieve the previous mask's network and subnetwork info from the DataFrame
            previous_mask_row = df.iloc[prev_mask_counter]
            previous_subnetwork = previous_mask_row['Subnetwork']
            previous_network = previous_mask_row['Network']

            # Record the overlap message
            overlapping_masks.append((
                current_network, current_subnetwork,
                previous_network, previous_subnetwork,
                overlap_count
            ))

    # Label the new mask (keep a copy of original data)
    labeled_mask = np.zeros_like(old_mask_data)
    labeled_mask[old_mask_data == 1] = mask_counter + 2

    # Update the template data
    temp_template_data += labeled_mask

    # Store the current original mask data for future overlap comparisons
    old_mask_data_list.append(old_mask_data)

# After running the loop, print the overlapping masks information
if overlapping_masks:
    print("Overlapping masks detected:")
    for current_network, current_subnetwork, previous_network, previous_subnetwork, count in overlapping_masks:
        # Calculate the cubic mm for the overlapping voxels
        overlap_volume = count * voxel_volume  # Total volume in cubic mm

        output_line = (
            f"Network: {current_network}, Subregions: {current_subnetwork} (Key: {current_subnetwork.split(':')[0]}) "
            f"overlap with Network: {previous_network}, Subregions: {previous_subnetwork} (Key: {previous_subnetwork.split(':')[0]}) "
            f"- Overlapping Voxel Count: {count}, Volume Overlap: {overlap_volume:.2f} mm³"
        )
        print(output_line)

        # Save the output line to a text file
        with open('/project/3013068.03/software/Core_Network_ROIs/overlapping_masks.txt', 'a') as f:
            f.write(output_line + '\n')

print("Overlapping masks information saved to 'overlapping_masks.txt'.")