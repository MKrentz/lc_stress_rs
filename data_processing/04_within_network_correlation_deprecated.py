import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import glob
import pingouin as pg
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for z-transformation


# Define session and run
session_list = ['control', 'stress']
run_list = [1, 2]
results = []  # Initialize a list to store results

# Iterate over sessions and runs
subject_list = [i[-7:] for i in glob.glob('/project/3013068.03/resting_state/sub*')]
exclusion_list = ['sub-006', 'sub-008', 'sub-010']
subject_list = [i for i in subject_list if i not in exclusion_list]


for sub_id in subject_list:
    for session in session_list:
        for run in run_list:
            # Load the DataFrame for the current subject, session, and run
            df = pd.read_pickle(
                f'/project/3013068.03/resting_state/{sub_id}/GS_extraction_output_{sub_id}_{session}_run-{run}.pkl')  # Load the DataFrame

            # Loop through each unique Network in the DataFrame
            for network in df['Network'].unique():
                # Extract Subnetwork data for the current network
                subnetworks_df = df[df['Network'] == network]

                # Collect timecourses into an array for correlation calculation
                timecourses = np.array(subnetworks_df['mean_timeseries'].tolist())

                # Ensure there are multiple subnetworks to compute correlation
                if timecourses.shape[0] > 1:
                    # Calculate the correlation matrix for the network
                    corr_matrix = np.corrcoef(timecourses)

                    # Calculate the mean correlation (upper triangle of the correlation matrix)
                    mean_corr = np.mean(corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)])

                    # Append results including subject, session, run
                    results.append({
                        'Subject': sub_id,
                        'Session': session,
                        'Run': run,
                        'Network': network,
                        'Mean_Correlation': mean_corr,
                    })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Z-transformation (standardization) of Mean_Correlation
scaler = StandardScaler()
results_df['Z_Mean_Correlation'] = scaler.fit_transform(results_df[['Mean_Correlation']])

# Now you can analyze differences across session-run combinations
# For example, pivot the DataFrame to compare Mean Correlation values across sessions and runs
comparison_df = results_df.pivot_table(index='Network', columns=['Session', 'Run'], values='Mean_Correlation')

# Iterate through each unique Subnetwork
anova_results = []

# Iterate through each unique Network
for network in results_df['Network'].unique():
    # Filter the DataFrame for the current Network
    subnetwork_df = results_df[results_df['Network'] == network]
    glob.glob(f'/project/3013068.03/software/Core_Network_ROIs/+{network}')

    # Check if there are enough observations to perform the ANOVA
    if subnetwork_df['Z_Mean_Correlation'].count() >= 2:  # Minimum observations needed
        # Perform repeated measures ANOVA
        anova = pg.rm_anova(data=subnetwork_df, dv='Z_Mean_Correlation', within=['Session', 'Run'], subject='Subject')

        # Check if the interaction effect is significant
        if 'Session * Run' in anova['Source'].values:
            interaction_p = anova.loc[anova['Source'] == 'Session * Run', 'p-unc'].values[0]

            if interaction_p < 0.05:
                # Perform post-hoc tests for multi-way interactions
                post_hoc = pg.pairwise_tests(data=subnetwork_df, dv='Z_Mean_Correlation',
                                              within=['Run', 'Session'], subject='Subject')
                post_hoc['Network'] = network  # Add network info to post-hoc results

                print(f"Post-hoc Tests for Network: {network}")
                print(post_hoc)

        # Store ANOVA results with network name
        anova_results.append({'Network': network, 'ANOVA_Table': anova})

# Convert results into a DataFrame if needed
anova_results_df = pd.DataFrame(anova_results)