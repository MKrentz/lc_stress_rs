# Data Preparation

This directory contains scripts for the spatial preprocessing and anatomical masking of resting-state fMRI data, prior to large-scale network connectivity analysis.

## Scripts

* **`prep_smooth_img.py`**
    Applies spatial smoothing to functional data across sessions and runs. 
    * Applies a 6mm FWHM kernel to MNI-space data.
    * Applies a 3mm FWHM kernel to native-space data.
    * Outputs are saved to the subject's `smoothed_imgs` directory.

* **`prep_ventricle_processing.py`**
    Prepares and refines 4th ventricle masks. 
    * Uses `fslmaths` (via `subprocess`) to smooth and threshold (`0.985`) existing CSF probability maps.
    * Binarizes and resamples the shrunk masks to match the resolution of the native functional data using nearest-neighbor interpolation.

* **`prep_ventricle_extraction.py`**
    Extracts the mean signal timecourse from the 4th ventricle.
    * Loads the resampled masks generated in the previous step.
    * Uses `nilearn.maskers.NiftiMasker` to extract and standardize the signal.
    * Saves the z-scored mean timecourses as `.csv` files to be used as nuisance regressors.

* **`submission.py`**
    Batch submission script for the HPC cluster (Torque/PBS/qsub). 
    * Automatically fetches the subject list from the project directory.
    * Filters out subjects based on a hardcoded exclusion list.
    * Submits jobs with predefined memory (`24000mb`) and walltime limits.

## Dependencies

* **Python packages:** `nilearn`, `nibabel`, `numpy`, `pandas`
* **External tools:** FSL (specifically `fslmaths` must be available in your environment path)
* **Local modules:** `Subject_Class_new` (handles subject data fetching)

## Usage Notes

Paths are currently hardcoded to the `/project/3013068.03/` base directory. If this repository is cloned or moved to a different environment, these paths will need to be updatd.
