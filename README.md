# Fetal MRI Image Generation using WGAN-GP

## Overview
This project implements a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to generate synthetic fetal MRI images. The model is trained on a dataset of fetal MRI scans and aims to produce high-quality synthetic images that resemble real medical scans. The implementation is done using PyTorch.

## Dataset
The dataset used for training is obtained from Kaggle:
[Nusrat Jahan Pritha - Fetal MRI Dataset](https://www.kaggle.com/datasets/nusratjahanpritha/fetal-mri)

### Steps to Download Dataset:
1. Upload your `kaggle.json` file to Colab.
2. Run the following commands in your notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   !mkdir -p ~/.kaggle
   !mv /content/kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json  # Set permissions
   
   !kaggle datasets download -d nusratjahanpritha/fetal-mri -p /content/data --unzip
   ```

## Model Architecture

### Generator
The generator network takes a random noise vector (latent space) of dimension 100 and transforms it into a 128x128x3 image through a series of transposed convolutional layers with batch normalization and ReLU activations.

### Discriminator
The discriminator is a convolutional neural network that classifies images as real or fake. It uses LeakyReLU activations and progressively downsamples the image using convolutional layers.

### Gradient Penalty
The model uses a gradient penalty term to stabilize training by ensuring the 1-Lipschitz constraint.

## Training Details
- **Epochs:** 500
- **Batch Size:** 8
- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Lambda GP:** 10
- **Critic Iterations:** 5 (Discriminator updates per Generator update)

## Training Process
1. Load the dataset and apply transformations (Resize, Normalize, and Convert to Tensor).
2. Train the Discriminator:
   - Compute loss for real and fake images.
   - Compute the gradient penalty.
   - Update the model parameters.
3. Train the Generator:
   - Generate fake images from random noise.
   - Compute loss based on discriminator feedback.
   - Update the model parameters.
4. Repeat steps for the specified number of epochs.
5. Save the generated images and model checkpoints every 10 epochs.

## Running the Model

### Requirements
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Google Colab (for training with GPU)

### Execution Steps
1. Clone the repository and navigate to the project directory.
2. Download the dataset as mentioned above.
3. Run the script in a Colab notebook or a local environment with GPU support.
4. The model will generate and save synthetic fetal MRI images every 10 epochs.

### Sample Code to Start Training
```python
# Ensure required directories exist
import os
os.makedirs("generated_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Start training
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        # Training loop (Discriminator & Generator)
```

## Model Checkpoints & Generated Images
- Model checkpoints are saved in `saved_models/`.
- Generated images are saved in `generated_images/`.

## Results & Observations
- The quality of generated images improves as training progresses.
- The gradient penalty helps in stabilizing the training process.
- The model is effective in generating synthetic fetal MRI images that resemble real medical scans.

## Future Improvements
- Increase batch size and training iterations for better results.
- Experiment with different GAN architectures (e.g., StyleGAN, Progressive GANs).
- Fine-tune hyperparameters to improve convergence.

## References
- **WGAN-GP Paper:** [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- **GAN Tutorial:** [PyTorch DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
