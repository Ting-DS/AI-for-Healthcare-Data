# AI-for-Healthcare-Nanodegree

#### [Certification Link](https://www.udacity.com/certificate/e/89d29302-de3f-11ee-bf00-9b008d3aab47)

<div align="center">
  <img src="https://github.com/Ting-DS/AI-for-Healthcare-Nanodegree/blob/main/certification.png" width="80%">
</div>

## Introduction
This repository contains 3 Healthcare Data Science projects using AI techniques created by [Ting Lu](https://www.linkedin.com/in/ting-lu-9949b0233/):

 - [2D Image Classification: Pneumonia Detection from Chest X-Rays](): Analyze data from the [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community), including 112,000 chest x-rays with disease labels acquired from 30,000 patients, and train a CNN with attention layer to predict the presence of pneumonia. Finally, prepare a FDA submission for [510(k) clearance](https://www.fda.gov/medical-devices/device-approvals-and-clearances/510k-clearances) as a medical device software. In addition to model development, the submission preparation will include model description, data introduction, and a validation plan compliant with FDA standards.

<div align="center">
  <img src="https://github.com/Ting-DS/Healthcare-Data-Science-AI/blob/main/Pneumonia-Detection-from-2D-Chest-X-Rays/inference_img.png" width="50%">
</div>

 - [3D Brain MRIs Segmentation: Quantifying Hippocampus Volume for Alzheimer's Progression](https://github.com/Ting-DS/Healthcare-Data-Science-AI/tree/main/Brain-MRIs-Segmentation-for-Alzheimer's-Progression): Built an end-to-end AI system including a [U-Net segmentation](https://towardsdatascience.com/understanding-u-net-61276b10f360) model that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients, as their studies are committed to the clinical imaging archive. Finally integrate the model into a working clinical PACS such that it runs on every incoming study and produces a report with volume measurements. The radiology department runs a HippoCrop tool which cuts out a rectangular portion of a brain scan from every MRI image series and annotated the relevant volumes, and converted them to NIFTI format.

![Hippocampus](https://github.com/Ting-DS/Healthcare-Data-Science-AI/blob/main/Brain-MRIs-Segmentation-for-Alzheimer's-Progression/readme_img/Hippocampus_small.gif)

 - [Diabetes Patient Selection for Clinical Trials using EHR data](https://github.com/Ting-DS/Healthcare-Data-Science-AI/tree/main/Clinical-Trial-Patient-Selection-using-EHR): Focus on a novel diabetes drug ready for clinical trials, administered over 5-7 days in hospitals with extensive monitoring and patient training via a mobile app. Utilize EHR data provided by a client partner to develop a deep learning regression model predicting hospitalization time, subsequently converted into a binary indicator for trial inclusion. Target patients likely requiring hospitalization for the drug's duration without significant additional costs. This project highlights the importance of precise data representation at the encounter level, involving filtering, preprocessing, and feature engineering of medical code sets, alongside analyzing and addressing biases across demographic groups in the model interpretation.

<div align="center">
  <img src="https://github.com/Ting-DS/Healthcare-Data-Science-AI/blob/main/Clinical-Trial-Patient-Selection-using-EHR/EHR.jpg" width="80%">
</div>
