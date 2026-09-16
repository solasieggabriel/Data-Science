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

![Comparison of Latent Space Separability](Comparison%20of%20Latent%20Space%20Separability.png)
The 2D projection plots show that Model B clearly separates the classes before quantum encoding, especially isolating Residential along the second principal component, while Model A's classes heavily overlap.


![Confusion Matrix Benchmark](Confusion%20Matrix%20Benchmark.png)
The confusion matrices reveal that Model B virtually eliminates the severe structural cross-confusion seen in Model A (e.g., 31 crop scenes and 22 water bodies misclassified as residential), achieving near-perfect 99.5% accuracy on Residential and confining minor residual errors solely to the natural optical boundary between dark water and dense forest.


![Training Convergence Dynamics](Training%20Convergence%20Dynamics.png)
The convergence dynamics demonstrate that Model B achieves both superior optimization efficiency and higher generalization stability, descending to a 68% lower loss and surpassing 90% accuracy within 4 epochs, while Model A exhibits noisy oscillations and plateaus at an 11% accuracy deficit.


### Head-to-Head Benchmark

| Metric | Model A (Raw Pixels + PCA) | Model B (8 Orthogonal Features) | Model B Advantage ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Final Test Accuracy** | 82.38% (659/800) | 93.38% (747/800) | +11.00% (+88 images correct) |
| **Final Training Loss** | 0.5904 | 0.1869 | -68.3% relative loss reduction |
| **Validation Loss** | 0.5875 | 0.1793 | -69.5% lower val loss |
| **Macro Average F1** | 0.8239 | 0.9339 | +0.1100 (+11.00%) |
| **Residential F1-Score** | 0.8205 (Prec: 76.86%) | 0.9950 (Prec: 99.50%) | +17.45% F1 (+22.64% Precision) |
| **AnnualCrop F1-Score** | 0.8649 (Recall: 80.00%) | 0.9723 (Recall: 96.50%) | +10.74% F1 (+16.50% Recall) |
| **SeaLake F1-Score** | 0.7688 (Recall: 74.00%) | 0.8728 (Recall: 87.50%) | +10.40% F1 (+13.50% Recall) |
| **Forest F1-Score** | 0.8413 (Recall: 87.50%) | 0.8955 (Recall: 90.00%) | +5.42% F1 (+2.50% Recall) |
| **Generalization Gap ($\epsilon_{\text{gen}}$)** | 0.87% | 0.10% | Good generalization |
| **Parameters** | 132 | 132 | Identical |
| **Epoch Duration** | ~105.3 s | ~105.4 s | Identical |

#### Key Findings
- Model B (8 orthogonal features) outperformed Model A (raw pixels + PCA) by +11.00% absolute test accuracy, correctly classifying 88 more test images.
- Model B beat Model A in every category, with the largest gains seen in structurally complex environments:
    - Residential: 99.50% F1 vs. 82.05% (+17.45% gain, achieving 99.5% precision and recall).
    - AnnualCrop: 97.23% F1 vs. 86.49% (+10.74% gain, with a +16.50% recall).
    - SeaLake: 87.28% F1 vs. 76.88% (+10.40% gain).
    - Forest: 89.55% F1 vs. 84.13% (+5.42% gain).
- Model B achieved a 68.3% relative reduction in training loss (0.1869 vs. 0.5904) and a 69.5% lower validation loss (0.1793 vs. 0.5875) under same epochs and runtime.

- Linear PCA could only capture 84.54% of total variance with 8 components while discarding the remaining 15.46% of detail as "noise", causing the model to plateau at 82.38% accuracy.
- Feeding four physical domains into the 8 qubits made each rotation angle $R_y(\theta_i)$ explore independent geometric axis, giving the 4-layer VQC much steeper loss gradients to separate classes.
- The +17.45% F1 jump in `Residential` accuracy proves that PCA blur out important details like street grids. Canny edge density and GLCM contrast preserved the distinct grid signals that PCA completely blurred out.
- Reaching 93.38% accuracy using only 132 total parameters (96 quantum and 36 classical) highlights how powerful quantum circuits can be when paired with informed, non-linear classical feature maps.

## Final Conclusion

When classifying complex Earth observation imagery under the strict qubit constraints of NISQ hardware (e.g., $n=8$), linear raw-pixel compression (PCA) hits a fundamental variance ceiling, severely degrading boundary resolution. In contrast, encoding a curated, mathematically orthogonal set of non-linear spatial, spectral, and statistical features gives better accuracy and drop in training loss, establishing domain-specific feature engineering as a superior, hardware-efficient paradigm for quantum computer vision.
