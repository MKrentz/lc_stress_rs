# Locus Coeruleus and Large-Scale Network Dynamics Under Acute Stress

This repository contains the data processing, physiological denoising, and statistical analysis pipeline for investigating the effects of acute stress on Locus Coeruleus (LC) functional connectivity and large-scale brain network reorganization.



## Overview

Acute stress triggers dynamic reconfigurations of large-scale brain networks, but the in-vivo role of the locus coeruleus-norepinephrine (LC-NE) system in these shifts is difficult to isolate in humans. This repository accompanies our study evaluating healthy adults (n=24) undergoing counterbalanced stress (SECPT) and control sessions. 

The pipeline combines individualized neuromelanin-sensitive MRI (for LC localization) with resting-state fMRI. 

## Repository Structure

The codebase is modularized into three main stages of the analysis pipeline. Each folder contains its own README explaining the specific scripts.

* **`data_preparation/`**
  Handles the initial spatial preprocessing of functional data. Includes scripts for applying differential spatial smoothing (native vs. MNI space) and refining 4th ventricle anatomical masks for nuisance signal extraction. Also contains cluster batch-submission scripts.

* **`orthogonalization/`**
  Contains part of the denoising pipeline. These scripts orthogonalize ICA-AROMA noise components against RETROICOR physiological confounds, isolating unique noise variance to aggressively regress out of the functional timeseries without removing shared neural variance.

* **`data_processing/`**
  The core analysis and statistical engine. Handles the final extraction of ROI timeseries, builds the comprehensive confound matrix, calculates Fisher Z-transformed functional connectivity matrices, and runs the group-level repeated-measures ANOVAs and post-hoc tests.

## Dependencies

The pipeline relies heavily on the Python ecosystem and standard statistical libraries, alongside FSL for specific mathematical operations on NIfTI files. 

**Main Python Packages:**
* `numpy`, `pandas`
* `nibabel`, `nilearn`
* `scipy`, `statsmodels`, `pingouin`
* `scikit-learn`
* `matplotlib`, `seaborn`

**External Software:**
* FSL (specifically `fslmaths` and `fsl_regfilt` must be available in your environment path)

## Getting Started & Usage Notes

1. **Environment Setup:** It is recommended to run this pipeline within a dedicated Conda or virtual environment to manage dependencies. 
2. **Path Management:** *Important:* The scripts in this repository were originally built for a specific HPC cluster environment. You will need to update the hardcoded base paths (e.g., `/project/.../` or `/home/user/.../`) to match your local directory structure or server mount points before running the scripts.
3. **Data Structure:** The pipeline assumes the input data has already been preprocessed via fMRIPrep and follows a standard BIDS-like directory structure for derivative data.

