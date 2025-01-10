# CSAI Bachelor Thesis

**Author:** Hamse Elmi (2023232)
**Supervisor:** Sasha Kenjeeva
**Second Reader:** Dr. Eriko Fukuda

## Overview

This repository contains the code for the Bachelor thesis titled "Early Diagnosis of Alzheimer's Using the Weighted Probability-Based Ensemble Method". The study explores the application of a Weighted Probability-Based Ensemble Method (WPBEM) for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Normal Cognition (NC) using Amyloid PET imaging data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI).

## Objectives

Early Diagnosis: Develop a robust method for early detection of Alzheimer's Disease using Amyloid PET scans.
Ensemble Learning: Implement the WPBEM by integrating DenseNet201, ResNet50, and VGG19 to enhance classification accuracy.
Comparative Analysis: Evaluate and compare WPBEM performance against individual CNN models.
Metrics Evaluation: Measure performance using accuracy, F1 score, AUC, sensitivity, and specificity.

## Dataset

The dataset used in this research is sourced from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), Phase 3 (ADNI3). The dataset comprises Amyloid PET scans that are preprocessed to extract relevant slices for deep learning applications.

## Methodology

**Data Preprocessing:** Conversion from DICOM to NIfTI, slice selection, normalisation, entropy-based prioritisation, and resizing to create a robust input dataset.
**Model Training:** Training three CNNs (DenseNet201, ResNet50, VGG19) on preprocessed slices to classify AD, MCI, and NC.
**Ensemble Learning:** Aggregation of predictions from the CNNs using WPBEM with weighted probabilities. (This part was taken bt Fathi et al., (2024?) and can be found at step 6 of the main code from line 366)
**Performance Evaluation:** Comparison of individual CNN and ensemble model results using diagnostic metrics.

## Folder Structure

Main Code

- code_final.py: Python file containing data preprocessing, model training, and evaluation.

Model Results

- Includes model results and comparative performance metrics for CNNs and WPBEM.

Scripts

- script_for_file_processing.ipynb: Jupyter notebook file containing script for file format conversion from DICOM to NIfTI.

## Requirements

- Python 3.x
- Libraries:
 ```
    numpy
    pandas
    matplotlib
    seaborn
    nibabel
    scikit-image
    scikit-learn
    pytorch
    torchvision
 ```
- Additional tools: dcm2niix: Conversion of DICOM files to NIfTI format.

## Acknowledgments

This research was conducted under the guidance of Sasha Kenjeeva. The data was obtained from the ADNI3 dataset, and the WPBEM framework was adapted from the work of Fathi et al. (2024). 
Contact

## For any questions or further information, please contact:
Hamse Elmi
Email: h.elmi@tilburguniversity.edu / hamse-elmi@hotmail.com
