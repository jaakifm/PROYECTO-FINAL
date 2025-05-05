# Computer Vision for Melanoma Detection

This folder contains various scripts and resources focused on the detection of melanomas in images. The goal is to develop and improve computer vision models to classify images as benign or malignant melanomas.


## Models Tested

We conducted extensive experiments with various architectures to achieve optimal melanoma classification performance:

### Initial Approach
- **Custom CNN**: At the beginning of the project, we developed a custom convolutional neural network from scratch to establish a baseline for melanoma detection. This model helped us understand the complexity of the task and the importance of feature extraction for skin lesion classification.

### Individual Models
- **ResNet50**: A deep residual network that allows training of deeper networks by introducing skip connections.
- **EfficientNetB0 with SE (Squeeze-and-Excitation)**: Enhanced EfficientNet with channel attention mechanisms to improve feature representation.
- **Vision Transformers (ViT)**:
  - **ViT-B16**: Base version of the Vision Transformer with 16x16 pixel patches.
  - **ViT-L16**: Larger version with increased capacity for feature learning.

### Ensemble Approach
We combined the best-performing models (ViT16 and SE-EfficientNetB0) into an ensemble. The optimal weight for each model's contribution was determined using various metaheuristic optimization techniques:

- **Genetic Algorithms**: Population-based optimization inspired by natural selection.
- **Simulated Annealing**: Probabilistic technique for approximating global optimization.
- **Particle Swarm Optimization**: Population-based optimization inspired by social behavior.
- **Grid Search**: Exhaustive search through specified parameter values.

This ensemble approach significantly improved classification performance compared to individual models.

## Datasets

The models in this folder have been primarily trained and evaluated using the Harvard HAM10000 dataset, a large collection of dermatoscopic images of common pigmented skin lesions. To ensure our models generalize well, we also performed validation with:

- **Kaggle Melanoma Classification Challenge Dataset**: Used for additional validation.

Cross-dataset evaluation helped confirm that our models can generalize to images captured under different conditions and with different equipment.

