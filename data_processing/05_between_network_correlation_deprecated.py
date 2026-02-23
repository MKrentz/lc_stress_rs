import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import glob
import pingouin as pg
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for z-transformation
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
import seaborn as sns

# Define session and run
session_list = ['control']
run_list = [1]
results = []  # Initialize a list to store results

# Iterate over sessions and runs
subject_list = [i[-7:] for i in glob.glob('/project/3013068.03/resting_state/sub*')]
exclusion_list = ['sub-006', 'sub-008', 'sub-010']
subject_list = [i for i in subject_list if i not in exclusion_list]

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)

dataframes = []
for sub_id in subject_list:
    for session in session_list:
        for run in run_list:
            # Load the DataFrame for the current subject, session, and run
            df = pd.read_pickle(
                f'/project/3013068.03/resting_state/{sub_id}/GS_extraction_output_{sub_id}_{session}_run-{run}.pkl')  # Load the DataFrame
            net_array = np.array(df['mean_timeseries'].tolist()).T
            #correlation_matrix = correlation_measure.fit_transform([net_array])[0]
            dataframes.append(net_array)
            '''
            # Example: Print the shape of each loaded DataFrame
            for i, df in enumerate(dataframes):
                print(f"Subject {i + 1}: {df.shape}")


            for i, df in enumerate(dataframes):
                array_3d[i] = df.values  # .values gives the underlying NumPy array of the DataFrame
            # Append results including subject, session, run
            contrast_matrix = np.zeros((51, 51))
            for i in range(51):
                for j in range(51):
                    if i != j:  # Don't compare self-correlation
                        if networks[i] == networks[j]:  # Same network
                            contrast_matrix[i, j] = 1  # Internal correlation
                            contrast_matrix[i, j] = -1  # External correlation
                        else:
                            contrast_matrix[i, j] = -1  # Internal correlation
                            contrast_matrix[i, j] = 1  # External correlation
            
            
            results.append({
                'Subject': sub_id,
                'Session': session,
                'Run': run,
                'Mean_Correlation': correlation_matrix,
            })
'''
corr_matrix = correlation_measure.fit_transform(np.array(dataframes))[1]
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap=plt.cm.coolwarm,
            square=True, linewidths=0.5)
plt.savefig('/project/3013068.03/testprint.png', format='png', bbox_inches='tight')
plt.close()
# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Z-transformation (standardization) of Mean_Correlation
scaler = StandardScaler()
results_df['Z_Mean_Correlation'] = scaler.fit_transform(results_df[['Mean_Correlation']])

# Now you can analyze differences across session-run combinations
# For example, pivot the DataFrame to compare Mean Correlation values across sessions and runs
comparison_df = results_df.pivot_table(index='Network', columns=['Session', 'Run'], values='Mean_Correlation')
