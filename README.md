# Low Light Image Enhancement Using GAN

## Overview

This project implements a Generative Adversarial Network (GAN) for enhancing low-light images. It uses a dataset of low-light and well-lit image pairs, and the GAN is trained to generate high-quality images from low-light input images. The model is built using PyTorch and can be used for applications in enhancing low-light images for computer vision tasks.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib
- skimage
- kaggle (for dataset)

To install the dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
torch
torchvision
Pillow
matplotlib
scikit-image
kaggle
```

## Setup

1. Download the dataset from Kaggle:

```bash
!kaggle datasets download -d soumikrakshit/lol-dataset
!unzip lol-dataset.zip -d /content/lol-dataset
```

2. Ensure the following directory structure:
    ```
    /content/lol-dataset/
        ├── low
        ├── high
        ├── eval15
    ```

## Training

### 1. Initialize Dataset
The dataset consists of pairs of low-light and well-lit images. The `LoLDataset` class loads these images for training.

### 2. Model Architecture
The project uses a GAN with the following components:
- **Generator**: Transforms low-light images to well-lit images.
- **Discriminator**: Differentiates between real and fake (generated) images.

### 3. Training Loop
The model is trained for 100 epochs, with the following steps:
- Train the discriminator using real and fake images.
- Train the generator to produce images that the discriminator classifies as real.

### 4. Evaluation
During training, the model is evaluated on the validation set using metrics such as SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio). The model performance is plotted after each epoch.

## Testing and Inference

After training, the model can be used to enhance low-light images. Sample images are passed through the trained generator, and the enhanced images are displayed alongside the input images.

## Results

After training, the following results can be observed:
- Visual comparison of input low-light images and the generated enhanced images.
- Performance metrics (SSIM and PSNR) for the validation set.

## Example Output

Here is a visualization of the enhanced images:

![Low-Light to Enhanced Images](path/to/your/images)

## Saving the Model

The trained generator model is saved to a file called `generator.pth` and can be loaded for inference.

```python
torch.save(generator.state_dict(), 'generator.pth')
```

To load the model for inference:

```python
generator.load_state_dict(torch.load('generator.pth'))
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **LOL Dataset**: [Low Light Image Enhancement Dataset (LOL)](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) used for training the model.
- **PyTorch**: The deep learning framework used for implementing the GAN.
- **Matplotlib**: For visualizing the results.

---
