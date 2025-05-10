## 📄 SMOTE-Diffusion: Data Augmentation for Imbalanced Datasets

![description](images/filename.png)

Reference Paper
SMOTE-Diffusion: An Over-Sampling Method Using Diffusion Models for Imbalanced Data
https://doi.org/10.48550/arXiv.2401.12345




### Overview
- **SMOTE-Diffusion** combines SMOTE (Synthetic Minority Over-sampling Technique) with **Diffusion Models** to generate synthetic samples for minority classes.
- **Goal**: Address class imbalance and enhance dataset diversity.
- **Applications**: IR-UWB radar object recognition, multi-sensor fusion, small-scale datasets.

### Repository Structure
- `smote_diffusion.py`: Core pipeline integrating SMOTE and diffusion model.
- `train.py`: Training script for the diffusion model.
- `generate.py`: Synthetic sample generation script.
- `evaluate.py`: Evaluation script to assess classification performance using augmented data.
- `dataset/`: Example datasets for experiments.

### Key Features
- Continuous latent space sampling using diffusion models.
- Captures complex data distributions beyond classical SMOTE.
- Generates data-driven synthetic samples via deep learning.

### Dependencies
```bash
python >=3.8
torch >=1.12
scikit-learn >=1.0
numpy




python train.py --epochs 100
python generate.py --num_samples 500
python evaluate.py --model resnet --dataset dataset/sample.csv




