# Orthogonalization & Denoising

This directory contains the pipeline for orthogonalizing ICA-AROMA noise components with respect to physiological confounds (RETROICOR) and regressing them out of the resting-state fMRI data. It also includes scripts to evaluate and visualize the variance explained by this process.

## Pipeline Scripts

The scripts are listed in their logical execution order:

* **`orthogonalization.py`**
    Isolates the unique noise variance by orthogonalizing the MELODIC/AROMA mixing matrices against the RETROICOR confounds. 
    * Computes the projection of the signal onto the base matrix using a pseudo-inverse and subtracts this projection to yield orthogonalized columns.
    
    * Automatically identifies noise components from AROMA output files.
    * Generates and saves pre- and post-orthogonalization intercorrelation heatmaps (`seaborn`) to verify the removal of shared variance.
    * Outputs the updated, orthogonalized mixing matrix as a `.tsv`.

* **`regfilt_settings.py`**
    Executes the actual data denoising using the orthogonalized matrices.
    * Dynamically generates and executes `fsl_regfilt` commands via `subprocess`.
    * Applies the aggressive or non-aggressive filtering to both MNI-space and native-space smoothed functional data.
    * Outputs the final cleaned NIfTI files to the `aroma_cleaned_data` directory.

* **`shared_variance_depiction.py`**
    Evaluates the impact of the orthogonalization process on the fMRI time series using a General Linear Model (GLM).
    * Fits two First-Level GLMs (`nilearn.glm`) to the functional data: one using the original mixing matrix and one using the orthogonalized matrix.
    * Computes z-score contrast maps for the variance explained by the noise predictors in both models.
    * Subtracts the two maps to calculate the spatial difference in variance explained, saving the output as `difference.nii.gz`.

* **`calculate_average_shared_variance.py`**
    Aggregates the individual GLM results for group-level visualization.
    * Loads the `difference.nii.gz` files across the entire filtered subject cohort.
    * Calculates the voxel-wise mean difference images.
    * Visualizes the final averaged difference map onto a standard brain template using `nilearn.plotting.plot_glass_brain` and saves it as a `.png`.
    

## Dependencies

* **Python packages:** `numpy`, `pandas`, `nibabel`, `nilearn`, `seaborn`, `matplotlib`
* **External tools:** FSL (specifically `fsl_regfilt` and `fslmaths`)
* **Local modules:** `Subject_Class_new`

## Usage Notes

* Similar to the data preparation pipeline, base paths are currently hardcoded to `/project/3013068.03/`. 
* Ensure the FSL module is properly loaded in your environment (the `regfilt_settings.py` script attempts to run `module load fsl`, which requires a compatible cluster environmet).
