# OcuNeuro OS

AI-assisted Stroke Risk Screening Prototype

## Overview

OcuNeuro OS is an educational research prototype that combines retinal vessel analysis from fundus images with clinical stroke-risk screening.

The system consists of two AI components:

1. Retinal AI
   - Uses a U-Net model trained on the FIVES retinal vessel segmentation dataset.
   - Produces a retinal vessel mask.
   - Calculates Vessel Density and Vessel Area.

2. Clinical AI
   - Uses Logistic Regression.
   - Estimates stroke-associated screening risk from selected clinical factors.

## Retinal AI Performance

Test set:
- 200 fundus images
- Dice Score: 88.54%
- IoU: 80.23%

## Clinical AI Performance

Test set:
- 1,022 samples
- ROC-AUC: 0.8438
- Sensitivity: 0.80
- Specificity: 0.7438

## Important Limitation

The retinal dataset and clinical stroke dataset contain different individuals.

Therefore, the current prototype does NOT claim to perform true multimodal fusion of retinal and clinical data. Retinal features and clinical risk are presented as separate outputs.

A true multimodal model would require matched fundus images, clinical variables, and stroke outcomes from the same patients.

## Disclaimer

This project is an educational and research prototype.

It is not a medical device and must not be used for diagnosis or clinical decision-making.
