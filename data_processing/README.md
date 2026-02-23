# Data Processing & Stats

This directory holds the main pipeline for extracting region-of-interest (ROI) timeseries, regressing out noise, and running the statistical analyses (connectivity and RM-ANOVA's).

## Scripts

* **`01_mask_creation.py`**
    Prepares and consolidates the network ROIs.
    * Maps the Shirer network ROIs to larger networks (Salience, Left/Right ECN, Ventral/Dorsal DMN).
    * Checks for spatial overlap between different masks to make sure they are mutually exclusive. Overlapping voxels are logged in `overlapping_masks.txt`.
    * Outputs a master dataframe (`network_df.csv`) with all ROI paths.

* **`02_rs_confound_creation.py`**
    Extracts the timeseries for each ROI and applies confound regression.
    * Assembles a confound matrix that includes Global Signal, CSF, White Matter, high-pass filter cosines, RETROICOR, and the 4th ventrical timecourses.
    * Uses `nilearn.maskers.NiftiMasker` to extract signals (from MNI space for the cortical networks, and native space for the Locus Coeruleus).
    * Calculates both the mean timeseries and the first eigenvariate (using PCA) for each region.
    * Saves everything as subject-specific `.pkl` files.

* **`03_correlation_stats_parametric_meta.py`**
    Executes the statistical testing and plotting.
    * Computes Pearson correlation matrices for the conditions (Stress/Control x Run 1/Run 2) and applies a Fisher Z-transformation.
    * Generates contrast matrices to isolate specific sub-networks (e.g., within-network, between-network, or network integrity).
    * Runs repeated-measures ANOVAs using `pingouin` to check for session and run effects, automatically running post-hoc Tukey tests if it finds an interaction.
    * Outputs the final plots (correlation grids, difference matrices, and bar plots with Cousineau-Morey corrected CIs).

## Dependencies

* **Data Handling & Stats:** `numpy`, `pandas`, `scipy`, `statsmodels`, `pingouin`, `scikit-learn`
* **Neuroimaging:** `nibabel`, `nilearn`
* **Visualization:** `matplotlib`, `seaborn`

## Usage Notes

* Like the other folders, base paths are currently hardcoded to the `/project/3013068.03/` directory.
* Please be awar of the `exclusion_list` (currently filtering out `sub-006`, `sub-008`, `sub-010`) if you add more subjects to the sample later.
