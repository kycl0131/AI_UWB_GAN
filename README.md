## SMOTE-Diffusion


<img src="main.png" width="700">


### Reference Paper
- **SMOTE-Diffusion: A Combined Approach for Authentic Data Generation for Time-Domain Radar Signal in Intelligent Transportation System**  
  [https://doi.org/10.1109/JSEN.2025.3544753](https://doi.org/10.1109/JSEN.2025.3544753)



## Overview

This repository implements **SMOTE-Diffusion**, a data augmentation method for stationary object detection using IR-UWB radar. To address the challenges of data scarcity and nonlinear signal distribution caused by position and angle variations, this approach combines SMOTE for initial data expansion with a Diffusion Model to generate nonlinear synthetic signals. This method improves data diversity and recognition performance.



## Introduction

Previous radar-based object detection research has mainly used FMCW radars, which are effective at detecting moving objects by leveraging Doppler information. However, FMCW radars struggle to detect stationary objects since Doppler information is unavailable for static targets. To overcome this limitation, IR-UWB radar has been employed, utilizing time-domain reflected signals for detection.

IR-UWB radar emits wideband pulses and receives the reflected signals in the time domain. Each time delay corresponds to the distance, reflection intensity, and surface properties of the object. Therefore, analyzing time-domain signals is essential for recognizing objects in real-world environments. However, IR-UWB radar signals are highly sensitive to variations in object position and angle, resulting in significant signal changes even for stationary objects. This leads to limited data availability, large data dispersion, and nonlinear data distribution.

Due to these characteristics, certain positions or angles may cause missing or unobserved signals, creating gaps in the data space. To compensate for insufficient training data and fill these gaps, a linear interpolation-based SMOTE technique was initially applied to predict missing signals. However, since SMOTE relies on linear interpolation, it failed to sufficiently capture the nonlinear and complex characteristics of IR-UWB radar signals, limiting the diversity and representational capacity of the generated data.

To address these limitations, this work proposes combining SMOTE for initial data space expansion with a Diffusion Model to generate nonlinear, high-quality synthetic signals. This approach mitigates both data scarcity and nonlinear distribution issues, ultimately improving data diversity and classification performance for stationary object detection with IR-UWB radar.


### Repository Structure


### Key Features
- Continuous latent space sampling using diffusion models.
- Captures complex data distributions beyond classical SMOTE.
- Generates data-driven synthetic samples via deep learning.

### Dependencies



python train.py --epochs 100
python generate.py --num_samples 500
python evaluate.py --model resnet --dataset dataset/sample.csv




