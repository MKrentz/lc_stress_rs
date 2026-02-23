import pandas as pd
import numpy as np
import glob
import os
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches

# Paths and settings
basepath = '/project/3013068.03/resting_state/'
output_path = '/project/3013068.03/resting_state/averaged_matrices/'

# Define session types and runs
session_types = ['control', 'stress']
runs = [1, 2]

# List subjects in directory
subjects = glob.glob(basepath + 'sub-*')
exclusion_list = ['sub-006', 'sub-008', 'sub-010']
subjects = [i for i in subjects if i[-7:]  not in exclusion_list]
# Initialize lists to store correlation matrices for each session-run combination


def fisher_z_transform(r):
    """ Apply Fisher's z transformation to a correlation coefficient. """
    return 0.5 * np.log((1 + r) / (1 - r))

def set_identity_diagonal(z_matrix, diagonal_value=1):
    """ Set the diagonal values to a specific diagonal_value (1 for identity). """
    # Use numpy's fill_diagonal to set all diagonal elements to diagonal_value
    np.fill_diagonal(z_matrix, diagonal_value)
    return z_matrix

def create_contrast_matrix(network_df, network_column, mode, network1=None, network2=None):
    """
    Create a contrast matrix based on the specified mode.

    Parameters:
        network_df (pd.DataFrame): DataFrame containing network information.
        network_column (str): The column in network_df that specifies the network.
        mode (str): The type of contrast matrix to create. Options: 'within', 'integrity', 'between'.
        network1 (str): The first network for comparison (required for 'within' and 'overlap' modes).
        network2 (str): The second network for comparison (required for 'overlap' mode).

    Returns:
        np.array: A contrast matrix based on the specified mode.
    """
    # Initialize the contrast matrix with zeros
    contrast_matrix = np.zeros((len(network_df['Subnetwork']), len(network_df['Subnetwork'])), dtype=int)

    # Iterate over pairs of Subnetworks to fill the contrast matrix
    for i, subnetwork_i in enumerate(network_df['Subnetwork']):
        network_i = network_df.loc[network_df['Subnetwork'] == subnetwork_i, network_column].values[0]
        for j, subnetwork_j in enumerate(network_df['Subnetwork']):
            network_j = network_df.loc[network_df['Subnetwork'] == subnetwork_j, network_column].values[0]

            if mode == 'within':
                # Set the positions representing overlap within the specified network to 1 (excluding diagonal)
                if (network_i == network1 and network_j == network1) and (i != j):  # Exclude diagonal
                    contrast_matrix[i, j] = 1

            elif mode == 'intergrity':
                # Mark within-network connections (excluding diagonal) with 1
                if (network_i == network1 and network_j == network1) and (i != j):
                    contrast_matrix[i, j] = 1

                # Mark connections between a region in the network and regions outside the network with -1
                if (network_i == network1 and network_j != network1) or (network_i != network1 and network_j == network1):
                    contrast_matrix[i, j] = -1

            elif mode == 'between':
                # Set the positions representing overlap between the two specified networks to 1
                if (network_i == network1 and network_j == network2) or (
                        network_i == network2 and network_j == network1):
                    contrast_matrix[i, j] = 1

    # Set values beyond the diagonal to 0
    for i in range(contrast_matrix.shape[0]):
        for j in range(i + 1, contrast_matrix.shape[1]):
            contrast_matrix[i, j] = 0  # Set upper triangle values to 0

    # Additional logic for 'integrity' mode
    if mode == 'integrity':
        # Explicitly set diagonal elements to 0
        np.fill_diagonal(contrast_matrix, 0)

    return contrast_matrix

def calculate_contrast(contrast_matrix, correlation_matrix, allow_negative=False):
    """
    Calculate the contrast value for a given contrast matrix and correlation matrix.

    Parameters:
    - contrast_matrix: A 2D numpy array indicating where overlaps (or contrasts) exist.
    - correlation_matrix: A 2D numpy array with correlations between subnetworks.
    - allow_negative: Boolean, if True define calculations for -1 values otherwise consider only 1 values.

    Returns:
    - result: Contrast result value, comparing mean correlations of designated matrix regions.
    """
    if allow_negative:
        # Calculate metrics for both within (value 1) and between (value -1) networks
        weighted_sum_within = np.sum(correlation_matrix * (contrast_matrix == 1))
        weighted_sum_between = np.sum(correlation_matrix * (contrast_matrix == -1))

        total_within = np.count_nonzero(contrast_matrix == 1)  # Count of 1s
        total_between = np.count_nonzero(contrast_matrix == -1)  # Count of -1s

        mean_within = weighted_sum_within / total_within if total_within > 0 else 0
        mean_between = weighted_sum_between / total_between if total_between > 0 else 0

        result = mean_within - mean_between

    else:
        # Calculate only for within-network (or overlap) regions defined by 1s
        weighted_sum_within = np.sum(correlation_matrix * (contrast_matrix == 1))
        total_within = np.count_nonzero(contrast_matrix == 1)

        mean_within = weighted_sum_within / total_within if total_within > 0 else 0

        # Since we don't have -1 to compare with, just output within-network mean
        result = mean_within

    return result

def calculate_within_subject_ci(df, dv, within_factors):
    """
    Calculate within-subject confidence intervals using the Cousineau-Morey method.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        dv (str): The dependent variable (e.g., 'z_correlation').
        within_factors (list): The within-subject factors (e.g., ['session', 'run']).

    Returns:
        pd.DataFrame: A dataframe with the mean and corrected confidence intervals.
    """
    # Normalize the data
    participant_means = df.groupby('sub_id')[dv].mean()  # Mean for each participant
    grand_mean = df[dv].mean()  # Grand mean across all data
    df_normalized = df.copy()
    df_normalized[dv] = df[dv] - participant_means[df['sub_id'].values].values + grand_mean

    # Calculate means and standard errors for each condition
    summary_df = df_normalized.groupby(within_factors)[dv].agg(
        mean=('mean'),
        sem=('sem')
    ).reset_index()

    # Apply Morey correction
    J = len(df[within_factors[0]].unique()) * len(df[within_factors[1]].unique())  # Number of conditions
    correction_factor = np.sqrt(J / (J - 1))
    summary_df['ci'] = summary_df['sem'] * correction_factor * 1.96  # 95% CI

    return summary_df

def plot_contrast_matrix(contrast_matrix, title, labels, filename):
    """
    Plots a contrast matrix as a heatmap with a binary colorbar for -1, 0, and 1.

    Parameters:
        contrast_matrix (np.array): The contrast matrix to plot.
        title (str): Title of the plot.
        labels (list): List of labels for the matrix axes.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))

    # Create a mask for the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(contrast_matrix, dtype=bool), k=1)

    # Define a custom colormap
    cmap = plt.cm.get_cmap('coolwarm', 3)  # 3 discrete colors
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries centered around -1, 0, 1
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)  # Normalize the colormap

    # Plot the heatmap
    sns.heatmap(
        contrast_matrix, mask=mask, cmap=cmap, norm=norm,
        square=True, cbar_kws={"shrink": 0.8, "ticks": [-1, 0, 1]},  # Set colorbar ticks
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, annot=False, fmt="d",  # Use integer formatting for annotations
        annot_kws={"size": 8}
    )

    # Customize the plot
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(corr_matrix, title, labels, filename, range_set=(-1,1)):
    """Plots the lower triangle of the correlation matrix as a heatmap and saves it as a PNG file."""
    # Add identity diagonal
    np.fill_diagonal(corr_matrix, 1)

    # Create a mask to cover the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 'k=1' excludes the diagonal from the mask

    plt.figure(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(
        corr_matrix, fmt=".2f", cmap="coolwarm",
        square=True, mask=mask, cbar_kws={"shrink": .8},
        xticklabels=labels, yticklabels=labels, linewidths=0.5,
        vmin=range_set[0], vmax=range_set[1],  # Symmetric color scale for difference matrices
        annot=True, annot_kws={"size": 8}  # Smaller font size for annotations
    )
    plt.title(title)

    # Rotate x-axis labels by 45 degrees and align them properly
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=8)  # Smaller font size for x-axis labels
    plt.yticks(fontsize=8)  # Smaller font size for y-axis labels
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()

'''
Not currently in use, as no single region comparisons are made.
def plot_p_matrix(corr_matrix, title, labels, filename):
    """Plots only the lower part of the correlation matrix as a heatmap with symbol annotations and saves it as a PNG file."""
    # Create a mask to cover the upper triangle and the diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  # 'k=0' includes the diagonal in the mask

    plt.figure(figsize=(12, 10))

    # Define a colormap for values from 0 to 1
    cmap = sns.heatmap

    # Prepare annotation matrix with symbols
    annotations = np.empty_like(corr_matrix, dtype=str)

    # Iterate only over the lower triangle
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1):  # Only annotate the lower triangle
            if corr_matrix[i, j] < 0.01:
                annotations[i, j] = '*'
            elif corr_matrix[i, j] < 0.05:
                annotations[i, j] = '+'
            else:
                annotations[i, j] = ''

    # Create the heatmap
    sns.heatmap(
        corr_matrix, fmt="", cmap="coolwarm",
        square=True, mask=mask, cbar_kws={"shrink": .8},
        xticklabels=labels, yticklabels=labels, linewidths=0.5,
        vmin=0, vmax=1
    )
    plt.title(title)

    # Save the figure
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
'''

def plot_correlation_matrices_in_grid(fisher_z_matrices, condition_names, labels, code_to_label, output_path, range_set=(-1, 1)):
    """Plots a 2x2 grid of correlation matrices with a shared color bar and a legend for coded labels."""
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=False, sharey=False, gridspec_kw={"wspace": 0.25, "hspace": 0.1})
    fig.subplots_adjust(right=0.8)  # Make space for the color bar and legend

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create a mask to cover the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones_like(fisher_z_matrices[list(fisher_z_matrices.keys())[0]], dtype=bool), k=1)

    # Plot each correlation matrix in the grid
    for counter, condition in enumerate(fisher_z_matrices):
        # Add identity diagonal
        corr_matrix = fisher_z_matrices[condition].copy()
        np.fill_diagonal(corr_matrix, 1)

        # Create the heatmap in the respective subplot
        sns.heatmap(
            corr_matrix, fmt=".2f", cmap="coolwarm",
            square=True, mask=mask, cbar=False,
            xticklabels=labels, yticklabels=labels, linewidths=0.5,
            vmin=range_set[0], vmax=range_set[1],
            annot=True, annot_kws={"size": 8},
            ax=axes[counter]
        )
        axes[counter].set_title(condition_names[counter], fontsize=22)
        axes[counter].tick_params(axis='both', labelsize=12)

        # Rotate x-axis labels by 45 degrees and align them properly
        plt.setp(axes[counter].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add a shared color bar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Position of the color bar
    fig.colorbar(axes[0].collections[0], cax=cbar_ax)

    # Add a legend for the coded labels
    legend_ax = fig.add_axes([0.95, 0, 0.1, 0.7])  # Position of the legend
    legend_ax.axis('off')  # Hide the axes

    # Create the legend
    # Create the legend with correct order (top to bottom)
    for idx, (code, label) in enumerate(code_to_label.items()):
        y_position = 1 - idx * 0.05  # Start from the top and move downward
        legend_ax.text(0, y_position, f"{code}: {label}", fontsize=14, va='top')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_path}fisher_z_matrices_grid.png', format='png', bbox_inches='tight')
    plt.close()

def plot_difference_matrices_in_grid(difference_matrices, labels, code_to_label, output_path, range_set=(-0.3, 0.3)):
    """
    Plots a 2x3 grid of difference matrices with a shared color bar and a legend for coded labels.

    Parameters:
        difference_matrices (dict): Dictionary of difference matrices with descriptive titles as keys.
        labels (list): List of labels for the matrix axes.
        code_to_label (dict): Dictionary mapping codes to labels for the legend.
        output_path (str): Path to save the output figure.
        range_set (tuple): Range for the color scale (default: (-0.3, 0.3)).
    """
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16), sharex=False, sharey=False,
                             gridspec_kw={"wspace": 0.3, "hspace": 0.2})
    fig.subplots_adjust(right=0.8)  # Make space for the color bar and legend

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create a mask to cover the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones_like(list(difference_matrices.values())[0], dtype=bool), k=1)

    # Plot each difference matrix in the grid
    for counter, (title, diff_matrix) in enumerate(difference_matrices.items()):
        # Create the heatmap in the respective subplot
        sns.heatmap(
            diff_matrix, fmt=".2f", cmap="coolwarm",
            square=True, mask=mask, cbar=False,
            xticklabels=labels, yticklabels=labels, linewidths=0.5,
            vmin=range_set[0], vmax=range_set[1],
            annot=True, annot_kws={"size": 8},
            ax=axes[counter]
        )
        axes[counter].set_title(title, fontsize=16)
        axes[counter].tick_params(axis='both', labelsize=10)

        # Rotate x-axis labels by 45 degrees and align them properly
        plt.setp(axes[counter].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add a shared color bar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Position of the color bar
    fig.colorbar(axes[0].collections[0], cax=cbar_ax)

    # Add a legend for the coded labels
    legend_ax = fig.add_axes([0.95, 0, 0.1, 0.7])  # Position of the legend
    legend_ax.axis('off')  # Hide the axes

    # Create the legend
    for idx, (code, label) in enumerate(code_to_label.items()):
        y_position = 1 - idx * 0.05  # Start from the top and move downward
        legend_ax.text(0, y_position, f"{code}: {label}", fontsize=12, va='top')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{output_path}difference_matrices_grid.png', format='png', bbox_inches='tight')
    plt.close()

def save_correlation_matrix_as_table(corr_matrix, title, labels, filename):
    """
    Saves the correlation matrix as a formatted table (lower triangle only) with real numbers.
    """
    # Create a mask for the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)

    # Mask the upper triangle and diagonal
    masked_matrix = np.where(mask, np.nan, corr_matrix)

    # Convert to DataFrame for pretty formatting
    df = pd.DataFrame(masked_matrix, index=labels, columns=labels)

    # Format numbers to 3 decimal places (adjust as needed)
    formatted_df = df.applymap(lambda x: f"{x:.3f}" if not np.isnan(x) else "")

    # Save to CSV (or .txt if preferred)
    formatted_df.to_csv(filename, sep='\t')  # Use \t for tab-separated values

    print(f"Saved formatted matrix to {filename}")



timeseries_df = pd.DataFrame(columns=['stress_1', 'stress_2', 'control_1', 'control_2'],
                              index=[os.path.basename(subject) for subject in subjects])

network_node_selection = ['Left Anterior Insula',
                          'ACC, MPFC, SMA',
                          'Right Anterior Insula',
                          'Left Middle Frontal Gyrus, Superior Frontal Gyrus',
                          'Left Inferior Frontal Gyrus, Orbitofrontal Gyrus',
                          'Left Superior Parietal Gyrus, Inferior Parietal Gyrus, Precuneus, Angular Gyrus',
                          'Right Middle Frontal Gyrus, Right Superior Frontal Gyrus',
                          'Right Middle Frontal Gyrus',
                          'Right Inferior Parietal Gyrus, Supramarginal Gyrus, Angular Gyrus',
                          'Medial Prefrontal Cortex, Anterior Cingulate Cortex, Orbitofrontal Cortex',
                          'Left Angular Gyrus',
                          'Posterior Cingulate Cortex, Precuneus',
                          'Right Angular Gyrus']
                          #'Locus Coeruleus']


network_shortcut = ['SN - 1',
                    'SN - 2',
                    'SN - 3',
                    'ECN - 1',
                    'ECN - 2',
                    'ECN - 3',
                    'ECN - 4',
                    'ECN - 5',
                    'ECN - 6',
                    'DMN - 1',
                    'DMN - 2',
                    'DMN - 3',
                    'DMN - 4']

network_node_selection_names = ['Left Anterior Insula - SN',
                          'ACC, MPFC, SMA - SN',
                          'Right Anterior Insula - SN',
                          'Left Middle Frontal Gyrus, Superior Frontal Gyrus - ECN',
                          'Left Inferior Frontal Gyrus, Orbitofrontal Gyrus - ECN',
                          'Left Superior Parietal Gyrus, Inferior Parietal Gyrus, Precuneus, Angular Gyrus - ECN',
                          'Right Middle Frontal Gyrus, Right Superior Frontal Gyrus - ECN',
                          'Right Middle Frontal Gyrus - ECN',
                          'Right Inferior Parietal Gyrus, Supramarginal Gyrus, Angular Gyrus - ECN',
                          'Medial Prefrontal Cortex, Anterior Cingulate Cortex, Orbitofrontal Cortex - DMN',
                          'Left Angular Gyrus - DMN',
                          'Posterior Cingulate Cortex, Precuneus - DMN',
                          'Right Angular Gyrus - DMN',]
                          #'Locus Coeruleus']

code_to_label = dict(zip(network_shortcut, network_node_selection_names))
correlation_measure = ConnectivityMeasure(
    kind="correlation"
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
                mask_df = mask_df[mask_df['Subnetwork'].isin(network_node_selection)]
                mask_df.drop(3, inplace=True)
                # Extract mean_timeseries and first_eigenvariate
                mean_timeseries = np.array(mask_df['mean_timeseries'].tolist())
                timeseries_df.loc[sub_id, f'{session}_{run}'] = mean_timeseries

all_corr = {'stress_1': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['stress_1'].to_list()),(0, 2, 1))),
            'stress_2': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['stress_2'].to_list()),(0, 2, 1))),
            'control_1': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['control_1'].to_list()),(0, 2, 1))),
            'control_2': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['control_2'].to_list()),(0, 2, 1)))}

condition_names = ['Stress - Run 1', 'Stress - Run 2', 'Control - Run 1', 'Control - Run 2']

network_df = pd.read_csv('/project/3013068.03/software/Core_Network_ROIs/network_df_meta.csv',
                         index_col = 0)
network_df.drop(13, inplace=True) #Exclude LC

labels = network_node_selection_names

fisher_z_matrices = {}
for condition in all_corr:
    fisher_z_matrices[condition] = np.mean([fisher_z_transform(matrix) for matrix in all_corr[condition]], axis=0)

stacked_array = np.stack([all_corr[key] for key in all_corr.keys()])
average_matrix = np.mean(stacked_array, axis=0)

difference_matrices = {
    'Stress Run 1 - Control Run 1': fisher_z_matrices['stress_1'] - fisher_z_matrices['control_1'],
    'Stress Run 1 - Control Run 2': fisher_z_matrices['stress_1'] - fisher_z_matrices['control_2'],
    'Stress Run 2 - Control Run 1': fisher_z_matrices['stress_2'] - fisher_z_matrices['control_1'],
    'Stress Run 2 - Control Run 2': fisher_z_matrices['stress_2'] - fisher_z_matrices['control_2'],
    'Stress Run 1 - Stress Run 2': fisher_z_matrices['stress_1'] - fisher_z_matrices['stress_2'],
    'Control Run 1 - Control Run 2': fisher_z_matrices['control_1'] - fisher_z_matrices['control_2'],
}
## ACTIVATE THIS TO ONLY GET IMPORTANT NODES
# Plot the difference matrices in a 2x3 grid
plot_difference_matrices_in_grid(
    difference_matrices=difference_matrices,
    labels=network_shortcut,
    code_to_label=code_to_label,
    output_path=output_path,
    range_set=(-0.2, 0.2)  # Adjust the range for difference matrices
)

df = pd.DataFrame(all_corr['stress_1'][0], index=network_df['Subnetwork'], columns=network_df['Subnetwork'])

# Iterate over pairs of Subnetworks to fill the contrast matrix
# Specify the network you want to analyze

'''
This section tests whether networks are consistently clustered by comparing mean correlation between network regions
with mean correlation between network regions and all other non-network regions. 
'''
network_list = ['Default Mode Network', 'Executive Control Network', 'Salience Network']

for network in network_list:
    # Create contrast matrix for the specified network using the new function
    conn_mat = create_contrast_matrix(network_df=network_df, network_column='sNetwork', mode='integrity',
                                      network1=network)

    t_list = []
    for i in range(average_matrix.shape[0]):
        # Apply Fisher Z transformation and set diagonal to identity
        fisher_z_scores = fisher_z_transform(average_matrix[i])
        fisher_z_scores = set_identity_diagonal(fisher_z_scores)

        # Calculate contrast and append to list
        t_list.append(calculate_contrast(conn_mat, fisher_z_scores, allow_negative=True))
        print(calculate_contrast(conn_mat, fisher_z_scores))

    # Perform one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(t_list, popmean=0)

    # Output the results
    print(f"T-statistic for {network}: {t_statistic}")
    print(f"P-value: {p_value}")

'''
This section calculated the mean correlation between two networks and subsequently adds subject-wise intercorrelation
to a long-format data frame for each subject, session, run, network-combination. 
Subsequently, RMANOVAs are calculated for all network-combinations.
'''
#network_list = ['anterior_Salience', 'post_Salience', 'LECN', 'RECN', 'ventral_DMN', 'dorsal_DMN', 'Locus Coeruleus']
network_list = ['Default Mode Network', 'Executive Control Network', 'Salience Network', 'Locus Coeruleus']

result_df = pd.DataFrame(columns=['sub_id', 'correlation', 'z_correlation', 'network_combination', 'session', 'run'])
# Generate all possible combinations of two networks
network_combinations = list(combinations(network_list, 2))
combination_list = []
# Print the resulting list of combinations
anova_results = []

for combination in combination_list:
    for session in session_types:
        for run in runs:
            # Create contrast matrix for the current network combination using the new function
            conn_mat = create_contrast_matrix(
                network_df=network_df,
                network_column='sNetwork',
                mode='between',
                network1=combination[0],
                network2=combination[1]
            )

            for sub_counter, cor_matrix in enumerate(all_corr[f'{session}_{run}']):
                # Apply Fisher Z transformation and set diagonal to identity
                fisher_z_scores = fisher_z_transform(cor_matrix)
                fisher_z_scores = set_identity_diagonal(fisher_z_scores)

                # Calculate raw and Z-transformed correlations
                sub_cor = calculate_contrast(conn_mat, cor_matrix)
                sub_cor_z = calculate_contrast(conn_mat, fisher_z_scores)

                # Append results to DataFrame
                new_row_df = [subjects[sub_counter], sub_cor, sub_cor_z, combination, session, run]
                result_df = pd.concat([result_df, pd.DataFrame([new_row_df], columns=result_df.columns)],
                                      ignore_index=True)

anova_results = []
posthoc_results = []

for network_combination in result_df['network_combination'].unique():
    # Filter the dataframe for this network combination
    df_subset = result_df[result_df['network_combination'] == network_combination]

    # Check assumption of normality
    normality_results = df_subset.groupby(['session', 'run']).apply(
        lambda x: pg.normality(x['z_correlation'])
    )
    print(f"\nNormality test results for network combination: {network_combination}")
    print(normality_results)

    # Check assumption of sphericity
    sphericity_test = pg.sphericity(data=df_subset, dv='z_correlation', within=['session', 'run'])
    print(f"Sphericity test results for network combination: {network_combination}")
    print(sphericity_test)

    # Proceed with RM-ANOVA only if normality and sphericity are met
    # Perform RM-ANOVA
    aovrm = pg.rm_anova(
        df_subset,
        dv='z_correlation',  # Dependent variable
        within=['session', 'run'],  # Within-subject factors
        subject='sub_id',  # Subject identifier
        detailed=True # Get detailed output
    )

    # Print or store the results
    print(f"\nANOVA results for network combination: {network_combination}")
    print(aovrm)

    # Keep results for later analysis
    anova_results.append((network_combination, aovrm))

    # Post Hoc Analysis using Pairwise Comparisons if interaction is significant
    interaction_p_value = aovrm.loc[aovrm['Source'] == 'session * run', 'p-unc'].values[0]

    if interaction_p_value is not None and interaction_p_value < 0.1:
        print(f"Significant interaction or trend found for network combination: {network_combination}")
        session_run_combination = df_subset['session'].astype(str) + '_' + df_subset['run'].astype(str)

        tukey_results = pairwise_tukeyhsd(endog=df_subset['z_correlation'], groups=session_run_combination,
                                          alpha=0.05)
        print("Post Hoc Tukey HSD results:")
        print(tukey_results)

        # Store post hoc results
        posthoc_results.append((network_combination, tukey_results))

        # Calculate within-subject confidence intervals
        summary_df = calculate_within_subject_ci(df_subset, dv='z_correlation', within_factors=['session', 'run'])
        print(summary_df)
        # Create bar plot with corrected confidence intervals
        plt.figure(figsize=(6, 7))
        control_patch = mpatches.Patch(color=(0.326, 0.618, 0.802), label="Control")
        stress_patch = mpatches.Patch(color=(0.837, 0.133, 0.130), label="Stress")

        # Plot bars for Control and Stress within each run
        for i, run in enumerate([1, 2]):
            control_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'control')]
            stress_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'stress')]

            # Plot Control bar
            plt.bar(
                i - 0.2, control_data['mean'].values[0], width=0.4,
                color=(0.32628988850442137, 0.6186236063052672, 0.802798923490965),
                edgecolor='black', label='Control' if i == 0 else ""
            )
            plt.errorbar(
                i - 0.2, control_data['mean'].values[0],
                yerr=control_data['ci'].values[0], fmt='none',
                c='black', capsize=5
            )

            # Plot Stress bar
            plt.bar(
                i + 0.2, stress_data['mean'].values[0], width=0.4,
                color=(0.8370472895040368, 0.13394848135332565, 0.13079584775086506),
                edgecolor='black', label='Stress' if i == 0 else ""
            )
            plt.errorbar(
                i + 0.2, stress_data['mean'].values[0],
                yerr=stress_data['ci'].values[0], fmt='none',
                c='black', capsize=5
            )
        sns.despine()
        plt.axhline(y=0, color='black', linewidth=1)
        # Set axis labels and title
        plt.title(f'{network_combination[0]} and {network_combination[1]}')
        plt.xlabel('Run')
        plt.xticks([0, 1], ['Run 1', 'Run 2'])
        plt.ylabel('Correlation')
        plt.ylim(ymin=-.3, ymax=.3)  # Adjust based on your data range

        # Show the legend
        plt.legend(title='Session', handles=[control_patch, stress_patch])

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'/project/3013068.03/resting_state/connectivity_effects/{network_combination}_trend_meta.png', dpi=300)
        plt.close()

    else:
        # Calculate within-subject confidence intervals
        summary_df = calculate_within_subject_ci(df_subset, dv='z_correlation', within_factors=['session', 'run'])
        print(summary_df)
        # Create bar plot with error bars
        # Create bar plot with corrected confidence intervals
        plt.figure(figsize=(6, 7))
        control_patch = mpatches.Patch(color=(0.326, 0.618, 0.802), label="Control")
        stress_patch = mpatches.Patch(color=(0.837, 0.133, 0.130), label="Stress")
        # Plot bars for Control and Stress within each run
        for i, run in enumerate([1, 2]):
            control_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'control')]
            stress_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'stress')]

            # Plot Control bar
            plt.bar(
                i - 0.2, control_data['mean'].values[0], width=0.4,
                color=(0.32628988850442137, 0.6186236063052672, 0.802798923490965),
                edgecolor='black', label='Control' if i == 0 else ""
            )
            plt.errorbar(
                i - 0.2, control_data['mean'].values[0],
                yerr=control_data['ci'].values[0], fmt='none',
                c='black', capsize=5
            )

            # Plot Stress bar
            plt.bar(
                i + 0.2, stress_data['mean'].values[0], width=0.4,
                color=(0.8370472895040368, 0.13394848135332565, 0.13079584775086506),
                edgecolor='black', label='Stress' if i == 0 else ""
            )
            plt.errorbar(
                i + 0.2, stress_data['mean'].values[0],
                yerr=stress_data['ci'].values[0], fmt='none',
                c='black', capsize=5
            )
        sns.despine()
        plt.axhline(y=0, color='black', linewidth=1)
        # Set axis labels and title
        plt.title(f'{network_combination[0]} and {network_combination[1]}')
        plt.xlabel('Run')
        plt.xticks([0, 1], ['Run 1', 'Run 2'])
        plt.ylabel('Correlation')
        plt.ylim(ymin=-.3, ymax=.3)  # Adjust based on your data range

        # Show the legend
        plt.legend(title='Session', handles=[control_patch, stress_patch])

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'/project/3013068.03/resting_state/connectivity_effects/{network_combination}_meta.png', dpi=300)
        plt.close()
        # ---- END OF BAR GRAPH PLOTTING ----

'''
This section tests whether the is a session, run or session*run effect for the average within-network correlation
'''

#network_list = ['anterior_Salience', 'post_Salience', 'LECN', 'RECN', 'ventral_DMN', 'dorsal_DMN']
network_list = ['Default Mode Network', 'Executive Control Network', 'Salience Network']

within_con_df = pd.DataFrame(columns=['sub_id', 'correlation', 'z_correlation', 'Network', 'session', 'run'])

for network in network_list:
    for session in session_types:
        for run in runs:
            # Create contrast matrix for the current network using the new function
            conn_mat = create_contrast_matrix(
                network_df=network_df,
                network_column='sNetwork',
                mode='within',
                network1=network
            )

            for sub_counter, cor_matrix in enumerate(all_corr[f'{session}_{run}']):
                # Calculate raw correlation
                sub_cor = calculate_contrast(conn_mat, cor_matrix)

                # Apply Fisher Z transformation and set diagonal to identity
                fisher_z_scores = fisher_z_transform(cor_matrix)
                fisher_z_scores = set_identity_diagonal(fisher_z_scores)

                # Calculate Z-transformed correlation
                sub_cor_z = calculate_contrast(conn_mat, fisher_z_scores)

                # Append results to DataFrame
                new_row_df = [subjects[sub_counter][-7:], sub_cor, sub_cor_z, network, session, run]
                within_con_df = pd.concat([within_con_df, pd.DataFrame([new_row_df], columns=within_con_df.columns)],
                                          ignore_index=True)

for network in within_con_df['Network'].unique():
    # Filter the dataframe for this network
    df_subset = within_con_df[within_con_df['Network'] == network]

    # Calculate within-subject confidence intervals
    summary_df = calculate_within_subject_ci(df_subset, dv='z_correlation', within_factors=['session', 'run'])

    # Create bar plot with corrected confidence intervals
    plt.figure(figsize=(6, 7))
    control_patch = mpatches.Patch(color=(0.326, 0.618, 0.802), label="Control")
    stress_patch = mpatches.Patch(color=(0.837, 0.133, 0.130), label="Stress")

    # Plot bars for Control and Stress within each run
    for i, run in enumerate([1, 2]):
        control_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'control')]
        stress_data = summary_df[(summary_df['run'] == run) & (summary_df['session'] == 'stress')]

        # Plot Control bar
        plt.bar(
            i - 0.2, control_data['mean'].values[0], width=0.4,
            color=(0.326, 0.618, 0.802),
            edgecolor='black', label='Control' if i == 0 else ""
        )
        plt.errorbar(
            i - 0.2, control_data['mean'].values[0],
            yerr=control_data['ci'].values[0], fmt='none',
            c='black', capsize=5
        )

        # Plot Stress bar
        plt.bar(
            i + 0.2, stress_data['mean'].values[0], width=0.4,
            color=(0.837, 0.133, 0.130),
            edgecolor='black', label='Stress' if i == 0 else ""
        )
        plt.errorbar(
            i + 0.2, stress_data['mean'].values[0],
            yerr=stress_data['ci'].values[0], fmt='none',
            c='black', capsize=5
        )

    sns.despine()
    plt.axhline(y=0, color='black', linewidth=1)

    # Set axis labels and title
    plt.title(f'Within-Network Connectivity for {network}', fontsize=14, pad=20)
    plt.xlabel('Run')
    plt.xticks([0, 1], ['Run 1', 'Run 2'])
    plt.ylabel('Mean Z-Correlation')
    plt.ylim(ymin=-.3, ymax=.8)  # Adjust based on your data range

    # Show the legend
    plt.legend(title='Session', handles=[control_patch, stress_patch])

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'/project/3013068.03/resting_state/connectivity_effects/{network}_within_connectivity.png', dpi=300)
    plt.close()

anova_results = []
posthoc_results = []
descriptives_results = []  # Store descriptives for each network

for network in within_con_df['Network'].unique():
    # Filter the dataframe for this network
    df_subset = within_con_df[within_con_df['Network'] == network]

    # Compute descriptive statistics before ANOVA
    descriptives_table = df_subset.groupby(['session', 'run']).agg(
        mean_correlation=('z_correlation', 'mean'),
        std_correlation=('z_correlation', 'std'),
        count=('z_correlation', 'count'),
        min_correlation=('z_correlation', 'min'),
        max_correlation=('z_correlation', 'max')
    ).reset_index()
    descriptives_table['Network'] = network
    descriptives_results.append(descriptives_table)
    # Print descriptive statistics
    print(f"Descriptive statistics for Network: {network}")
    print(descriptives_table)
    print(descriptives_table['mean_correlation'])

    # Conduct ANOVA
    aovrm = pg.rm_anova(
        df_subset,
        dv ='z_correlation',
        subject ='sub_id',
        within=['session', 'run'],
        detailed=True  # Ensure mean is calculated for the ANOVA
    )
    anova_table = aovrm

    # Print or store the results
    print(f"ANOVA results for network: {network}")
    print(anova_table)

    # Keep results for later analysis
    anova_results.append((network, anova_table))

    # Post Hoc Analysis using Pairwise Comparisons if interaction is significant
    interaction_p_value = aovrm.loc[aovrm['Source'] == 'session * run', 'p-unc'].values[0]

    if interaction_p_value is not None and interaction_p_value < 0.05:
        print(f"Significant interaction found for network: {network}")
        session_run_combination = df_subset['session'].astype(str) + '_' + df_subset['run'].astype(str)

        tukey_results = pairwise_tukeyhsd(endog=df_subset['z_correlation'], groups=session_run_combination, alpha=0.05)
        print("Post Hoc Tukey HSD results:")
        print(tukey_results)

        # Store post hoc results
        posthoc_results.append((network, tukey_results))
