import pandas as pd
import numpy as np
import glob
import os
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import zscore
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap, BoundaryNorm


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

def plot_p_matrix(corr_matrix, title, labels, filename, setting=[-1, 1]):
    """Plots only the lower part of the correlation matrix as a heatmap and saves it as a PNG file."""
    # Create a mask to cover the upper triangle and the diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  # 'k=0' includes the diagonal in the mask

    plt.figure(figsize=(12, 10))

    # Define a custom colormap and normalize based on specified boundaries
    colors = ['green', 'yellow', 'red']
    cmap = ListedColormap(colors)
    boundaries = [0, 0.05, 0.1, 1]  # Define edges of the intervals
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Create the heatmap
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap=cmap,
                norm=norm, square=True, mask=mask, cbar_kws={"shrink": .8},
                xticklabels=labels, yticklabels=labels, linewidths=0.5)
    plt.title(title)

    # Save the figure
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()



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

def create_contrast_matrix(network_df, specified_network, network_column):
    contrast_matrix = np.zeros((len(network_df['Subnetwork']), len(network_df['Subnetwork'])), dtype=int)
    # Iterate over pairs of Subnetworks to fill the contrast matrix
    for i, subnetwork_i in enumerate(network_df['Subnetwork']):
        network_i = network_df.loc[network_df['Subnetwork'] == subnetwork_i, network_column].values[0]
        for j, subnetwork_j in enumerate(df.columns):
            network_j = network_df.loc[network_df['Subnetwork'] == subnetwork_j, network_column].values[0]
            if i == j:
                continue  # Skip diagonal elements as they are not needed.
            # Count if both are in the specified network ('LECN')
            if network_i == specified_network and network_j == specified_network:
                contrast_matrix[i, j] = 1  # Both belong to the specified network

            # Check for -1 condition
            elif (network_i != network_j) and (network_i == specified_network or network_j == specified_network):
                contrast_matrix[i, j] = -1  # One of the Subnetworks belongs to the specified network

    # Set values beyond the diagonal to 0
    for i in range(contrast_matrix.shape[0]):
        for j in range(i + 1, contrast_matrix.shape[1]):
            contrast_matrix[i, j] = 0  # Set upper triangle values to 0

    # Identify valid rows and columns
    valid_rows = np.any(contrast_matrix == 1, axis=1)  # Mask for rows containing at least one 1
    valid_cols = np.any(contrast_matrix == 1, axis=0)  # Mask for columns containing at least one 1

    # Zero out invalid rows and columns
    for i in range(contrast_matrix.shape[0]):
        if not valid_rows[i]:  # If the row does not contain a 1
            for j in range(contrast_matrix.shape[1]):
                if not valid_cols[j]:
                    contrast_matrix[i, j] = 0
    return contrast_matrix

def create_overlap_contrast_matrix(network_df, network1, network2, network_column):
    # Initialize the contrast matrix with zeros
    contrast_matrix = np.zeros((len(network_df['Subnetwork']), len(network_df['Subnetwork'])), dtype=int)

    # Iterate over pairs of Subnetworks to fill the contrast matrix
    for i, subnetwork_i in enumerate(network_df['Subnetwork']):
        network_i = network_df.loc[network_df['Subnetwork'] == subnetwork_i, network_column].values[0]
        for j, subnetwork_j in enumerate(network_df['Subnetwork']):
            network_j = network_df.loc[network_df['Subnetwork'] == subnetwork_j, network_column].values[0]

            # Set the positions representing overlap between the two specified networks to 1
            if (network_i == network1 and network_j == network2) or (
                    network_i == network2 and network_j == network1):
                contrast_matrix[i, j] = 1

    # Set values beyond the diagonal to 0
    for i in range(contrast_matrix.shape[0]):
        for j in range(i + 1, contrast_matrix.shape[1]):
            contrast_matrix[i, j] = 0  # Set upper triangle values to 0
    return contrast_matrix

def create_eigen_contrast_matrix(network_df, network, network_column):
    contrast_matrix = np.zeros((len(network_df['Subnetwork']), len(network_df['Subnetwork'])), dtype=int)

    # Iterate over pairs of Subnetworks to fill the contrast matrix
    for i, subnetwork_i in enumerate(network_df['Subnetwork']):
        network_i = network_df.loc[network_df['Subnetwork'] == subnetwork_i, network_column].values[0]
        for j, subnetwork_j in enumerate(network_df['Subnetwork']):
            network_j = network_df.loc[network_df['Subnetwork'] == subnetwork_j, network_column].values[0]

            # Set the positions representing overlap between the two specified networks to 1
            if (network_i == network and network_j == network):
                contrast_matrix[i, j] = 1

    # Set values beyond the diagonal to 0
    for i in range(contrast_matrix.shape[0]):
        for j in range(i + 1, contrast_matrix.shape[1]):
            contrast_matrix[i, j] = 0  # Set upper triangle values to 0
    return contrast_matrix

correlation_measure = ConnectivityMeasure(
    kind="correlation"
)

timeseries_df = pd.DataFrame(columns=['stress_1', 'stress_2', 'control_1', 'control_2'],
                              index=[os.path.basename(subject) for subject in subjects])

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
                timeseries_df.loc[sub_id, f'{session}_{run}'] = mean_timeseries

all_corr = {'stress_1': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['stress_1'].to_list()),(0, 2, 1))),
            'stress_2': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['stress_2'].to_list()),(0, 2, 1))),
            'control_1': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['control_1'].to_list()),(0, 2, 1))),
            'control_2': correlation_measure.fit_transform(np.transpose(np.array(timeseries_df['control_2'].to_list()),(0, 2, 1)))}

stacked_array = np.stack([all_corr[key] for key in all_corr.keys()])
average_matrix = np.mean(stacked_array, axis=0)

## ACTIVATE THIS TO GET FINE GRAINED RESULTS FOR ALL NETWORK NODES
network_df = pd.read_csv('/project/3013068.03/software/Core_Network_ROIs/network_df.csv',
                         index_col = 0)
## ACTIVATE THIS TO ONLY GET IMPORTANT NODES
network_df = pd.read_csv('/project/3013068.03/software/Core_Network_ROIs/network_df_meta.csv',
                         index_col = 0)
network_node_selection = ['Left Anterior Insula',
                          'ACC, MPFC, SMA',
                          'Right Anterior Insula',
                          'Left Middle Frontal Gyrus, Superior Frontal Gyrus',
                          'Left Inferior Frontal Gyrus, Orbitofrontal Gyrus',
                          'Left Superior Parietal Gyrus, Inferior Parietal Gyrus, Precuneus, Angular Gyrus',
                          'Right Middle Frontal Gyrus, Right Superior Frontal Gyrus',
                          'Right Middle Frontal Gyrus_2',
                          'Right Inferior Parietal Gyrus, Supramarginal Gyrus, Angular Gyrus',
                          'Medial Prefrontal Cortex, Anterior Cingulate Cortex, Orbitofrontal Cortex',
                          'Left Angular Gyrus',
                          'Posterior Cingulate Cortex, Precuneus',
                          'Right Angular Gyrus',
                          'Locus Coeruleus']


df = pd.DataFrame(all_corr['stress_1'][0], index=network_df['Subnetwork'], columns=network_df['Subnetwork'])

# Iterate over pairs of Subnetworks to fill the contrast matrix
# Specify the network you want to analyze

'''
This section tests whether networks are consistently clustered by comparing mean correlation between network regions
with mean correlation between network regions and all other non-network regions. 
'''
#network_list = ['anterior_Salience', 'post_Salience', 'LECN', 'RECN', 'ventral_DMN', 'dorsal_DMN']
network_list = ['Default Mode Network', 'Executive Control Network', 'Salience Network']

for network in network_list:
    conn_mat = create_contrast_matrix(network_df=network_df, specified_network=network, network_column='sNetwork')
    t_list = []
    for i in range(average_matrix.shape[0]):
        fisher_z_scores = fisher_z_transform(average_matrix[i])
        fisher_z_scores = set_identity_diagonal(fisher_z_scores)
        t_list.append(calculate_contrast(conn_mat, fisher_z_scores, allow_negative=True))
        print(calculate_contrast(conn_mat, fisher_z_scores))
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

for combination in network_combinations:
    combination_list.append(combination)

for combination in combination_list:
    for session in session_types:
        for run in runs:
            conn_mat = create_overlap_contrast_matrix(network_df,
                                                      network1=combination[0],
                                                      network2=combination[1],
                                                      network_column='sNetwork')
            for sub_counter, cor_matrix in enumerate(all_corr[f'{session}_{run}']):
                fisher_z_scores = fisher_z_transform(cor_matrix)
                fisher_z_scores = set_identity_diagonal(fisher_z_scores)
                sub_cor = calculate_contrast(conn_mat, cor_matrix)
                sub_cor_z = calculate_contrast(conn_mat, fisher_z_scores)
                new_row_df = [subjects[sub_counter], sub_cor, sub_cor_z, combination, session, run]
                result_df.loc[len(result_df)] = new_row_df

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
    aovrm = AnovaRM(
        df_subset,
        'z_correlation',
        'sub_id',
        within=['session', 'run'],
        aggregate_func='mean'  # Ensure mean is calculated for the ANOVA
    )
    anova_table = aovrm.fit()

    # Print or store the results
    print(f"\nANOVA results for network combination: {network_combination}")
    print(anova_table)

    # Keep results for later analysis
    anova_results.append((network_combination, anova_table))

    # Post Hoc Analysis using Pairwise Comparisons if interaction is significant
    interaction_p_value = anova_table.anova_table['Pr > F'].get('session:run', None)

    if interaction_p_value is not None and interaction_p_value < 0.1:
        print(f"Significant interaction or trend found for network combination: {network_combination}")
        session_run_combination = df_subset['session'].astype(str) + '_' + df_subset['run'].astype(str)

        tukey_results = pairwise_tukeyhsd(endog=df_subset['z_correlation'], groups=session_run_combination,
                                          alpha=0.05)
        print("Post Hoc Tukey HSD results:")
        print(tukey_results)

        # Store post hoc results
        posthoc_results.append((network_combination, tukey_results))

        summary_df = df_subset.groupby(['session', 'run']).agg(
            mean_correlation=('correlation', 'mean'),
            sem_correlation=('correlation', lambda x: stats.sem(x))
        ).reset_index()
        print(summary_df)
        summary_df['position'] = [0, 1, 2, 3]

        # Create bar plot with error bars
        plt.figure(figsize=(6, 12))

        # Use the adjusted positions for plotting
        sns.barplot(data=summary_df, x='position', y='mean_correlation', hue='session',
                    edgecolor='black',
                    palette={'control': 'blue', 'stress': 'orange'},
                    errorbar=None,)  # ci=None to avoid seaborn calculating confidence intervals


        index_list = [0, 1, 2, 3]
        for index, row in summary_df.iterrows():
            plt.errorbar(index_list[index], row['mean_correlation'],
                         yerr=row['sem_correlation'], fmt='none',
                         c='black', capsize=5)

        # Set axis labels and title
        plt.title(f'{network_combination}')
        plt.xlabel('Session and Run')
        plt.xticks([0, 1, 2, 3], ['Control - Run 1', 'Control - Run 2', 'Stress - Run 1', 'Stress - Run 2'])
        plt.ylabel('Correlation')
        plt.ylim(ymin=-.3, ymax=.3)  # Adjust based on your data range

        # Show the legend
        plt.legend(title='Session')

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'/project/3013068.03/resting_state/connectivity_effects/{network_combination}_trend.png')
        plt.close()

    else:
        summary_df = df_subset.groupby(['session', 'run']).agg(
            mean_correlation=('correlation', 'mean'),
            sem_correlation=('correlation', lambda x: stats.sem(x))
        ).reset_index()
        summary_df['position'] = [0, 1, 2,  3]

        # Create bar plot with error bars
        plt.figure(figsize=(6, 12))
        print(summary_df)
        # Use the adjusted positions for plotting
        sns.barplot(data=summary_df, x='position', y='mean_correlation', hue='session',
                    edgecolor='black',
                    palette={'control': 'blue', 'stress': 'orange'},
                    errorbar=None)  # ci=None to avoid seaborn calculating confidence intervals

        index_list = [0, 1, 2, 3]
        for index, row in summary_df.iterrows():
            plt.errorbar(index_list[index], row['mean_correlation'],
                         yerr=row['sem_correlation'], fmt='none',
                         c='black', capsize=5)

        # Set axis labels and title
        plt.title(f'{network_combination}')
        plt.xlabel('Session and Run')
        plt.xticks([0, 1, 2, 3], ['Control - Run 1', 'Control - Run 2', 'Stress - Run 1', 'Stress - Run 2'])
        plt.ylabel('Correlation')
        plt.ylim(ymin=-.3, ymax=.3)  # Adjust based on your data range

        # Show the legend
        plt.legend(title='Session')

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'/project/3013068.03/resting_state/connectivity_effects/{network_combination}.png')
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
            conn_mat = create_eigen_contrast_matrix(network_df, network, 'sNetwork')
            for sub_counter, cor_matrix in enumerate(all_corr[f'{session}_{run}']):
                sub_cor = calculate_contrast(conn_mat, cor_matrix)
                fisher_z_scores = fisher_z_transform(cor_matrix)
                fisher_z_scores = set_identity_diagonal(fisher_z_scores)
                sub_cor_z = calculate_contrast(conn_mat, fisher_z_scores)
                new_row_df = [subjects[sub_counter][-7:], sub_cor, sub_cor_z, network, session, run]
                within_con_df.loc[len(within_con_df)] = new_row_df


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
    aovrm = AnovaRM(
        df_subset,
        'z_correlation',
        'sub_id',
        within=['session', 'run'],
        aggregate_func='mean'  # Ensure mean is calculated for the ANOVA
    )
    anova_table = aovrm.fit()

    # Print or store the results
    print(f"ANOVA results for network: {network}")
    print(anova_table)

    # Keep results for later analysis
    anova_results.append((network, anova_table))

    # Post Hoc Analysis using Pairwise Comparisons if interaction is significant
    interaction_p_value = anova_table.anova_table['Pr > F'].get('session:run', None)

    if interaction_p_value is not None and interaction_p_value < 0.05:
        print(f"Significant interaction found for network: {network}")
        session_run_combination = df_subset['session'].astype(str) + '_' + df_subset['run'].astype(str)

        tukey_results = pairwise_tukeyhsd(endog=df_subset['z_correlation'], groups=session_run_combination, alpha=0.05)
        print("Post Hoc Tukey HSD results:")
        print(tukey_results)

        # Store post hoc results
        posthoc_results.append((network, tukey_results))

'''
Pairwise analysis of each cell with FDR-correction
'''
# Assume all_corr is a dictionary with keys: 'control_1', 'control_2', 'stress_1', 'stress_2'
# and each associated value is a list of correlation matrices for each subject.

# Determine the size of one of the correlation matrices
matrix_shape = all_corr['control_1'][0].shape

# Initialize matrices to store the FDR corrected p-values for each effect
fdr_corrected_p_matrices = {
    'session': np.zeros(matrix_shape),
    'run': np.zeros(matrix_shape),
    'session:run': np.zeros(matrix_shape)
}

p_matrices = {
    'session': np.zeros(matrix_shape),
    'run': np.zeros(matrix_shape),
    'session:run': np.zeros(matrix_shape)
}
# Store raw p-values separately for each effect
p_values = {
    'session': {},
    'run': {},
    'session:run': {}
}
index_pairs = []

# Iterate over the lower triangle of the matrix excluding the diagonal
for i in range(1, matrix_shape[0]):
    for j in range(i):

        # Prepare data for ANOVA
        data = {
            'subject': [],
            'session': [],
            'run': [],
            'value': []
        }

        conditions = [('control', '1'), ('control', '2'), ('stress', '1'), ('stress', '2')]

        for session, run in conditions:
            condition_key = f'{session}_{run}'
            for sub_id, matrix in enumerate(all_corr[condition_key]):
                # Append subject id, session, run, and specific correlation value
                fisher_z_scores = fisher_z_transform(matrix)
                fisher_z_scores = set_identity_diagonal(fisher_z_scores) #sets the inf diagonal to 1
                data['subject'].append(sub_id)
                data['session'].append(session)
                data['run'].append(run)
                data['value'].append(fisher_z_scores[i, j])

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Perform repeated-measures ANOVA
        aovrm = AnovaRM(df, 'value', 'subject', within=['session', 'run']).fit()

        # Extract p-values for the main effects and interaction
        p_vals = aovrm.anova_table['Pr > F'].values
        p_values['session'][(i, j)] = p_vals[0]
        p_values['run'][(i, j)] = p_vals[1]
        p_values['session:run'][(i, j)] = p_vals[2]
        index_pairs.append((i, j))

'''
FDR-Corrected
'''
# Apply FDR correction to the p-values for each effect
for effect in ['session', 'run', 'session:run']:
    p_values_list = [p_values[effect][idx] for idx in index_pairs]
    reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')

    # Fill the matrix with FDR-corrected p-values in the correct locations (lower triangle)
    for idx, corrected_p_value in zip(index_pairs, pvals_corrected):
        fdr_corrected_p_matrices[effect][idx] = corrected_p_value

    # Print significant results
    significant_results = [(i, j, fdr_corrected_p_matrices[effect][i, j]) for i, j in index_pairs if
                           fdr_corrected_p_matrices[effect][i, j] < 0.05]
    print(f"Significant p-values for {effect} below 0.05:")
    for result in significant_results:
        print(f"Cell ({result[0]}, {result[1]}) - FDR-corrected p-value: {result[2]}")

# Example: Access and print the FDR-corrected p-value matrices
for effect in ['session', 'run', 'session:run']:
    print(f"FDR-corrected p-values matrix for {effect}:")
    print(fdr_corrected_p_matrices[effect])
    plot_p_matrix(fdr_corrected_p_matrices[effect],
              title=f'BH-FDR corrected p-map for {effect} Effect',
              labels=network_df['Subnetwork'],
              filename=f'/project/3013068.03/resting_state/p_map_{effect}.png',
              setting = [0,1])

'''
Uncorrected
'''
# Apply FDR correction to the p-values for each effect
for effect in ['session', 'run', 'session:run']:
    p_values_list = [p_values[effect][idx] for idx in index_pairs]

    # Fill the matrix with FDR-corrected p-values in the correct locations (lower triangle)
    for idx, p_value in zip(index_pairs, p_values_list):
        p_matrices[effect][idx] = p_value

    # Print significant results
    significant_results = [(i, j, p_matrices[effect][i, j]) for i, j in index_pairs if
                           p_matrices[effect][i, j] < 0.05]
    print(f"Significant p-values for {effect} below 0.05:")
    for result in significant_results:
        print(f"Cell ({result[0]}, {result[1]}) - FDR-corrected p-value: {result[2]}")

# Example: Access and print the FDR-corrected p-value matrices
for effect in ['session', 'run', 'session:run']:
    print(f"P-values matrix for {effect}:")
    print(p_matrices[effect])
    plot_p_matrix(p_matrices[effect],
              title=f'Uncorrected p-map for {effect} Effect',
              labels=network_df['Subnetwork'],
              filename=f'/project/3013068.03/resting_state/p_map_{effect}_uncorrected.png',
              setting = [0,1])

'''
Only LC
'''
matrix_shape = all_corr['control_1'][0].shape

# Initialize matrices to store the FDR corrected p-values for each effect
fdr_corrected_p_matrices = {
    'session': np.zeros(matrix_shape),
    'run': np.zeros(matrix_shape),
    'session:run': np.zeros(matrix_shape)
}

p_matrices = {
    'session': np.zeros(matrix_shape),
    'run': np.zeros(matrix_shape),
    'session:run': np.zeros(matrix_shape)
}

# Store raw p-values separately for each effect
p_values = {
    'session': {},
    'run': {},
    'session:run': {}
}
index_pairs = []

# Use the last row index for the iteration
last_row_index = matrix_shape[0] - 1

# Iterate over all columns of the last row (excluding the diagonal)
for j in range(last_row_index):

    # Prepare data for ANOVA
    data = {
        'subject': [],
        'session': [],
        'run': [],
        'value': []
    }

    conditions = [('control', '1'), ('control', '2'), ('stress', '1'), ('stress', '2')]

    for session, run in conditions:
        condition_key = f'{session}_{run}'
        for sub_id, matrix in enumerate(all_corr[condition_key]):
            # Append subject id, session, run, and specific correlation value
            fisher_z_scores = fisher_z_transform(matrix)
            fisher_z_scores = set_identity_diagonal(fisher_z_scores) # Ensure diagonal is handled appropriately
            data['subject'].append(sub_id)
            data['session'].append(session)
            data['run'].append(run)
            data['value'].append(fisher_z_scores[last_row_index, j])  # Fixed row, iterate over columns

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Perform repeated-measures ANOVA
    aovrm = AnovaRM(df, 'value', 'subject', within=['session', 'run']).fit()

    # Extract p-values for the main effects and interaction
    p_vals = aovrm.anova_table['Pr > F'].values
    p_values['session'][(last_row_index, j)] = p_vals[0]
    p_values['run'][(last_row_index, j)] = p_vals[1]
    p_values['session:run'][(last_row_index, j)] = p_vals[2]
    index_pairs.append((last_row_index, j))

# Apply FDR correction to the p-values for each effect
for effect in ['session', 'run', 'session:run']:
    p_values_list = [p_values[effect][idx] for idx in index_pairs]
    reject, pvals_corrected, _, _ = multipletests(p_values_list, alpha=0.05, method='fdr_bh')

    # Fill the matrix with FDR-corrected p-values in the correct locations (last row)
    for idx, corrected_p_value in zip(index_pairs, pvals_corrected):
        fdr_corrected_p_matrices[effect][idx] = corrected_p_value

    # Print significant results
    significant_results = [(i, j, fdr_corrected_p_matrices[effect][i, j]) for i, j in index_pairs if
                           fdr_corrected_p_matrices[effect][i, j] < 0.05]
    print(f"Significant p-values for {effect} below 0.05:")
    for result in significant_results:
        print(f"Cell ({result[0]}, {result[1]}) - FDR-corrected p-value: {result[2]}")

    descriptive_stats = []

    for idx, corrected_p_value in zip(index_pairs, pvals_corrected):
        if corrected_p_value < 0.3 and effect == 'session:run':  # Focus on interaction effect
            # Retrieve original data for descriptive statistics
            interaction_data = {
                "Mean": [],
                "StdDev": [],
                "Count": []
            }
            for session, run in conditions:
                session_data = df[(df['session'] == session) & (df['run'] == run)]['value']
                interaction_data["Mean"].append(session_data.mean())
                interaction_data["StdDev"].append(session_data.std())
                interaction_data["Count"].append(session_data.count())

            # Collect stats
            descriptive_stats.append({
                'Cell': idx,
                'Corrected p-value': corrected_p_value,
                'Descriptive Statistics': interaction_data
            })

        # Output or save descriptive statistics
    print(f"Descriptive statistics for significant interaction effects (p < 0.05): {effect}")
    for stats in descriptive_stats:
        print(f"Cell: {stats['Cell']}, Corrected p-value: {stats['Corrected p-value']}")
        print(f"Descriptive Stats: {stats['Descriptive Statistics']}")