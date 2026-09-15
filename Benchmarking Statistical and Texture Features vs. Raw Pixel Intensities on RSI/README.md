# Benchmarking Statistical and Texture Features vs Raw Pixel Intensities on Remote Sensing Imagery

This study benchmarks the performance of an 8-qubit Variational Quantum Classifier (VQC) on remote sensing imagery (RSI), comparing the efficacy of classical dimensionality reduction against handcrafted feature engineering.

## Problem Statement
In the Noisy Intermediate-Scale Quantum (NISQ) era, encoding high-resolution imagery directly into quantum circuits faces severe hardware limitations, such as barren plateaus, limited qubit counts, and shallow circuit depth restrictions. This project investigates whether embedding domain-specific statistical and texture features outperforms unsupervised linear dimensionality reduction (PCA) of raw pixel intensities, holding the quantum circuit architecture strictly identical.

## Dataset and Methodology
The experiment utilizes the Sentinel-2 Land Cover dataset (EuroSAT_RGB) consisting of 64x64 pixel RGB images.
* **Classes**: The dataset includes a balanced sample of 4000 images across four distinct categories (1000 images for each): SeaLake, Forest, Residential, AnnualCrop.
* **Data Splitting**: The data is stratified into an 80% training set (3200 samples) and a 20% testing set (800 samples) using a fixed random seed.
* **Model A (Raw Pixels)**: Raw image arrays are flattened and compressed down to exactly 8 principal components using PCA.
* **Model B (Engineered Features)**: Extracts 8 specific domain features: Mean Blue, Intensity Standard Deviation, Grayscale Skewness, Green-Red Spectral Index (GRSI), GLCM Contrast, GLCM Energy, GLCM Correlation, Canny Edge Density.

## Model Architecture
The hybrid model integrates classical preprocessing with high-performance state-vector quantum simulation.
* **Quantum Circuit**: An 8-qubit parameterized circuit featuring `AngleEmbedding` and 4 layers of `StronglyEntanglingLayers`.
* **Measurement and Optimization**: Utilizes all-qubit Pauli-Z measurements fed into a classical linear projection head to map to the 4 output classes.
* **Frameworks**: The environment relies on PennyLane for quantum simulation and adjoint differentiation, PyTorch for classical backpropagation, scikit-learn for PCA, and skikit-image alongside OpenCV for spatial metrics.

---

## Summary of Results

