import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nilearn.connectome import ConnectivityMeasure

def plot_lower_correlation_matrix(corr_matrix, title, labels, filename, network_labels, setting =[-1, 1]):
    """Plots only the lower part of the correlation matrix as a heatmap and saves it as a PNG file."""
    # Create a mask to cover the upper triangle and the diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    plt.figure(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap=plt.cm.coolwarm,
                square=True, mask=mask, vmin=setting[0], vmax=setting[1],
                cbar_kws={"shrink": .8}, linewidths=0.5,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)

    # Get unique network labels and their indices
    unique_networks = np.unique(network_labels)

    # Draw rectangles for each network, but only for the lower triangle
    '''
    for network in unique_networks:
        # Get indices of columns and rows that belong to the current network
        indices = np.where(np.array(network_labels) == network)[0]

        # Ensure indices are only used for the lower triangle
        for i in indices:
            for j in indices:
                if i > j:  # Only include those lower than the diagonal
                    rect = Rectangle((j, i), 1, 1, linewidth=0.5, edgecolor='black', facecolor='none')
                    plt.gca().add_patch(rect)
    '''
    '''
    for i, network in enumerate(unique_networks):
        # Position vertically centered relative to the subnetworks
        idx_pos = np.where(network_labels == network)[0]
        if len(idx_pos) > 0:
            y_pos = (idx_pos[0] + idx_pos[-1]) / 2  # Center position of the network block
            x_pos = (idx_pos[0] + idx_pos[-1]) / 2 + 10
            plt.text(x_pos, y_pos, network, fontsize=12, va='bottom', ha='right', fontweight='bold')
    '''
    # Save the figure
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()

# Paths and settings
basepath = '/project/3013068.03/resting_state/'
output_path = '/project/3013068.03/resting_state/averaged_matrices/'
os.makedirs(output_path, exist_ok=True)

# Define session types and runs
session_types = ['control', 'stress']
runs = [1, 2]

# List subjects in directory
subjects = glob.glob(basepath + 'sub-*')
# Initialize lists to store correlation matrices for each session-run combination
mean_corr_matrices = {f"{session}_{run}": [] for session in session_types for run in runs}
eigen_corr_matrices = {f"{session}_{run}": [] for session in session_types for run in runs}

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)


# Loop through subjects and calculate correlation matrices
for subject in subjects:
    sub_id = os.path.basename(subject)
    print(sub_id)
    for session in session_types:
        for run in runs:
            # Load the respective .pkl file
            file_path = f'{basepath}{sub_id}/GS_extraction_output_{sub_id}_{session}_run-{run}.pkl'
            if os.path.exists(file_path):
                mask_df = pd.read_pickle(file_path)

                # Extract mean_timeseries and first_eigenvariate
                mean_timeseries = np.array(mask_df['mean_timeseries'].tolist())
                first_eigenvariates = np.array(mask_df['first_eigenvariate'].tolist())

                # Calculate and store correlation matrices
                mean_corr_matrices[f"{session}_{run}"].append(mean_timeseries)
                eigen_corr_matrices[f"{session}_{run}"].append(first_eigenvariates)

                corr_matrix_timeseries = correlation_measure.fit_transform([mean_timeseries.T])[0]
                corr_matrix_first_eigenvariate = correlation_measure.fit_transform([first_eigenvariates.T])[0]
                plot_lower_correlation_matrix(corr_matrix=corr_matrix_timeseries,
                                          labels=mask_df['Subnetwork'],
                                          title=f'Correlation Matrix for Mean {sub_id} {session}, Run {run})',
                                          filename=f'{basepath}{sub_id}/GS_mean_corr_plot_{sub_id}_{session}_run-{run}.png',
                                              network_labels=mask_df['Network'])

                plot_lower_correlation_matrix(corr_matrix=corr_matrix_first_eigenvariate,
                                          labels=mask_df['Subnetwork'],
                                          title=f'Correlation Matrix for First Eigenvariate {sub_id} {session}, Run {run})',
                                          filename=f'{basepath}{sub_id}/GS_mean_eigenvariat_plot_{sub_id}_{session}_run-{run}.png',
                                              network_labels=mask_df['Network'])

# Average correlation matrices for each session-run combination
averaged_mean_matrices = {}
averaged_eigen_matrices = {}
difference_matrices = {}

for run in runs:
    control_key = f"control_{run}"
    stress_key = f"stress_{run}"
    stress_mat = correlation_measure.fit_transform(np.transpose(np.array(mean_corr_matrices[f'stress_{run}']), (0,2,1)))[1]
    control_mat = correlation_measure.fit_transform(np.transpose(np.array(mean_corr_matrices[f'control_{run}']), (0,2,1)))[1]
    diff_mat = stress_mat - control_mat
    # Plot and save the difference matrix
    file_path = f"{output_path}GS_diff_corr_matrix_stress_control_run_{run}.png"
    plot_lower_correlation_matrix(corr_matrix=diff_mat,
                                  labels=mask_df['Subnetwork'],  # Assuming the labels are still relevant
                                  title=f'Difference Correlation Matrix (Stress - Control) for Run {run}',
                                  filename=file_path,
                                  network_labels=mask_df['Network'],
                                  setting=[-1, 1])  # Adjust the setting range according to your data

for session in session_types:
    for run in runs:
        corr_mat = \
        correlation_measure.fit_transform(np.transpose(np.array(mean_corr_matrices[f'{session}_{run}']), (0, 2, 1)))[1]
        corr_mat_eigen = correlation_measure.fit_transform(np.transpose(np.array(eigen_corr_matrices[f'{session}_{run}']), (0, 2, 1)))[1]

        file_path = f"{output_path}GS_averaged_corr_matrix_mean_{session}_{run}.png"
        plot_lower_correlation_matrix(corr_matrix=corr_mat,
                                      labels=mask_df['Subnetwork'],
                                      title=f'Average Correlation Matrix for Mean Timeseries ({session}_{run})',
                                      filename=file_path,
                                      network_labels=mask_df['Network'])

        file_path = f"{output_path}GS_averaged_corr_matrix_eigen_{session}_{run}.png"
        plot_lower_correlation_matrix(corr_matrix=corr_mat_eigen,
                                      labels=mask_df['Subnetwork'],
                                      title=f'Average Correlation Matrix for Eigenvariate Timeseries ({session}_{run})',
                                      filename=file_path,
                                      network_labels=mask_df['Network'])
for key in mean_corr_matrices.keys():
    averaged_mean_matrices[key] = np.mean(mean_corr_matrices[key], axis=0)
    averaged_eigen_matrices[key] = np.mean(eigen_corr_matrices[key], axis=0)

# Plot and save the averaged matrices for mean timeseries
for id, matrix in averaged_mean_matrices.items():
    file_path = f"{output_path}GS_averaged_corr_matrix_mean_{id}.png"
    plot_lower_correlation_matrix(corr_matrix=matrix,
                                  labels=mask_df['Subnetwork'],
                                  title=f'Average Correlation Matrix for Mean Timeseries ({id})',
                                  filename=file_path,
                                  network_labels=mask_df['Network'])

# Plot and save the averaged matrices for first eigenvariate
for id, matrix in averaged_eigen_matrices.items():
    file_path = f"{output_path}GS_averaged_corr_matrix_eigen_{id}.png"
    plot_lower_correlation_matrix(corr_matrix=matrix,
                                  labels=mask_df['Subnetwork'],
                                  title=f'Average Correlation Matrix for First Eigenvariate ({id})',
                                  filename=file_path,
                                  network_labels=mask_df['Network'])


var_mean_corr_matrices = {f"{session}_{run}": [] for session in session_types for run in runs}
var_eigen_corr_matrices = {f"{session}_{run}": [] for session in session_types for run in runs}

# Calculate variance matrices for mean correlation matrices
for key in mean_corr_matrices.keys():
    var_mean_corr_matrices[key] = np.var(mean_corr_matrices[key], axis=0)
    var_eigen_corr_matrices[key] = np.var(eigen_corr_matrices[key], axis=0)

# Plot and save the variance matrices for mean timeseries
for id, matrix in var_mean_corr_matrices.items():
    file_path = f"{output_path}GS_var_corr_matrix_mean_{id}.png"
    plot_lower_correlation_matrix(corr_matrix=matrix,
                                  labels=mask_df['Subnetwork'],
                                  title=f'Variance of Correlation Matrix for Mean Timeseries ({id})',
                                  filename=file_path,
                                  network_labels=mask_df['Network'],
                                  setting=[-.3, .3])

# Plot and save the variance matrices for first eigenvariate
for id, matrix in var_eigen_corr_matrices.items():
    file_path = f"{output_path}GS_var_corr_matrix_eigen_{id}.png"
    plot_lower_correlation_matrix(corr_matrix=matrix,
                                  labels=mask_df['Subnetwork'],
                                  title=f'Variance of Correlation Matrix for First Eigenvariate ({id})',
                                  filename=file_path,
                                  network_labels=mask_df['Network'],
                                  setting=[-.3, .3])