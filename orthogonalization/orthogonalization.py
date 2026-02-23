import numpy as np
import pandas as pd
import glob
from Subject_Class_new import Subject
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend, which is non-GUI
import matplotlib.pyplot as plt

def project_onto_a(b: pd.Series, a: pd.DataFrame):
    """Projects vector b onto the column space of matrix A."""
    # Ensure b is a 2D array (n, 1)
    b = b.values.reshape(-1, 1)
    # Pseudo-inverse of A
    pseudo_inverse_a = np.linalg.pinv(a.values)
    # Perform the dot products
    projection = a.dot(pseudo_inverse_a.dot(b)).values.flatten()  # Ensure the result is flattened
    return pd.Series(projection)  # Return as Series


def orthogonalize_columns(signal_matrix: pd.DataFrame, base_matrix: pd.DataFrame, columns_to_orthogonalize: list) -> pd.DataFrame:
    """Orthogonalizes specified columns in the signal matrix with respect to the base matrix."""
    modified_signal_matrix = signal_matrix.copy()

    for col in columns_to_orthogonalize:
        signal_column = modified_signal_matrix.iloc[:, col]  # Access the column as a Series
        proj = project_onto_a(signal_column, base_matrix)  # Project onto the base matrix
        modified_signal_column = signal_column - proj  # Orthogonalize the column
        modified_signal_matrix.iloc[:, col] = modified_signal_column  # Update the modified column

    return modified_signal_matrix

part_list = [i[-7:] for i in glob.glob('/project/3013068.03/resting_state/sub*')]
session_list = [1, 2]
run_list = [1, 2]

def plot_correlation_matrices(retro_df: pd.DataFrame, mm_array: np.ndarray, ortho_mm_df: pd.DataFrame, noise_list: list,
                              file_path: str):
    """Plot the lower half of the intercorrelation matrices and save the plot to a file with labeled axes."""

    # Calculate correlation matrices
    corr_before = np.corrcoef(retro_df.T, mm_array.T)  # Correlation before orthogonalization
    corr_after = np.corrcoef(retro_df.T, ortho_mm_df.T)  # Correlation after orthogonalization

    # Create a figure for the plots, increase size for readability
    plt.figure(figsize=(16, 10))

    # Define labels based on the specified rules
    retro_labels = list(retro_df.columns)  # Use existing labels from retro_df
    mm_labels = []
    ortho_labels = []

    # Construct labels for mm_array and ortho_mm_df based on noise_list
    for index in range(mm_array.shape[1]):
        if index in noise_list:
            mm_labels.append(f'Noise {index + 1}')
        else:
            mm_labels.append(f'Signal {index + 1}')

    for index in range(ortho_mm_df.shape[1]):
        if index in noise_list:
            ortho_labels.append(f'Noise {index + 1}')
        else:
            ortho_labels.append(f'Signal {index + 1}')

    # Plot for before orthogonalization
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(corr_before, mask=np.triu(np.ones_like(corr_before, dtype=bool)),
                cmap='coolwarm', center=0, annot=False, square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Before Orthogonalization')

    # Combine labels for the x-axis and y-axis
    all_labels_before = retro_labels + mm_labels
    # Set x-ticks with adjusted positions to shift slightly to the right
    plt.xticks(ticks=np.arange(len(all_labels_before)) + 0.5, labels=all_labels_before, rotation=90, fontsize=8,
               ha='center')
    # Set y-ticks with adjusted positions to shift slightly down
    plt.yticks(ticks=np.arange(len(all_labels_before)) + 0.5, labels=all_labels_before, fontsize=8, ha='right')
    # Plot for after orthogonalization
    ax2 = plt.subplot(1, 2, 2)
    sns.heatmap(corr_after, mask=np.triu(np.ones_like(corr_after, dtype=bool)),
                cmap='coolwarm', center=0, annot=False, square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix After Orthogonalization')

    # Combine labels for the x-axis and y-axis for after orthogonalization
    all_labels_after = retro_labels + ortho_labels
    # Set x-ticks with adjusted positions
    plt.xticks(ticks=np.arange(len(all_labels_after)) + 0.5, labels=all_labels_after, rotation=90, fontsize=8,
               ha='center')
    plt.yticks(ticks=np.arange(len(all_labels_after)) + 0.5, labels=all_labels_after, fontsize=8, ha='right')
    # Adjust the layout to make sure labels are not overlapping
    plt.tight_layout(pad=3.0)

    # Save the figure to the specified file path
    plt.savefig(file_path, format='png')
    plt.close()

exclusion_list = ['sub-006', 'sub-008', 'sub-010']
part_list = [i for i in part_list if i not in exclusion_list]

# Iterate through each subject in the list of parts
for part in part_list:
    # Create an instance of the Subject class for the current subject
    sub = Subject(part)

    # Iterate through sessions
    for session in session_list:
        # Iterate through runs
        for run in run_list:
            # Load the mixing matrix from a TSV file for the current subject, session, and run
            mm_df = pd.read_csv(glob.glob(f'/project/3013068.03/fmriprep_test/{part}/ses-mri0{session + 1}/func/'
                                          f'{part}_ses-mri0{session + 1}_*E28RS*run-{run}_desc-MELODIC_mixing.tsv')[0],
                                delimiter='\t',  # Specify tab delimiter
                                header=None)  # No header present in the file

            # Obtain retroicor confounds for the current run and session from the Subject instance
            retro_df = sub.get_retroicor_confounds(run=run, session=session)

            # Load the AROMA noise components from a CSV file
            noise_df = pd.read_csv(glob.glob(f'/project/3013068.03/fmriprep_test/{part}/ses-mri0{session + 1}/func/'
                                             f'{part}_ses-mri0{session + 1}_*E28RS*run-{run}_AROMAnoiseICs.csv')[0],
                                   header=None)

            # Extract the first row of the noise DataFrame as a list and adjust indexing (subtracting 1)
            noise_list = noise_df.iloc[0].tolist()  # Convert first row to a list
            noise_list = [i - 1 for i in noise_list]  # Adjust from 1-based to 0-based indexing

            # Orthogonalize the mixing matrix using the retroicor confounds and adjusted noise list
            ortho_mm_df = orthogonalize_columns(mm_df, retro_df, noise_list)

            # Define the file path for saving the correlation plot based on the run and session
            plot_file_path = f'/project/3013068.03/resting_state/{part}/ortho_mm/correlation_plot_run{run}_session{session}.png'

            # Call the function to plot correlation matrices and save the plot to the specified file path
            plot_correlation_matrices(retro_df, mm_df, ortho_mm_df, noise_list, plot_file_path)
            print(f'Created Corr-Matrix plot for {part} Session {session} Run {run}')
            # Save the orthogonalized mixing matrix to a CSV file
            ortho_mm_df.to_csv(
                f'/project/3013068.03/resting_state/{part}/ortho_mm/orthogonalized_mm_run-{run}_session{session}.tsv',
                index=False,
                header=False,
                sep ='\t'
            )

