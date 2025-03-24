# Image Classifier Project

This project is part of the Udacity AI Programming with Python Nanodegree. It's an image classification application that can train on a dataset and predict the class of an image.

## Project Overview

The project involves building a command-line application that allows users to:
- Train a neural network on a dataset of images
- Use a trained network to predict the class for an input image

The model uses a pre-trained network (VGG16 or VGG13) that is then customized with a new classifier suitable for the flower classification task.

## Files in the Repository

- `train.py`: Command line application for training the neural network
- `predict.py`: Command line application for making predictions using a trained model

## How to Use

### Training a Model

```bash
python train.py data_directory --save_dir checkpoint.pth --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

Arguments:
- `data_dir`: Directory with the training data (required)
- `--save_dir`: Path to save the checkpoint (default: 'checkpoint.pth')
- `--arch`: Model architecture, either 'vgg16' or 'vgg13' (default: 'vgg16')
- `--learning_rate`: Learning rate for training (default: 0.001)
- `--hidden_units`: Number of hidden units in the classifier (default: 512)
- `--epochs`: Number of training epochs (default: 5)
- `--gpu`: Use GPU for training if available (flag)

### Making Predictions

```bash
python predict.py input_image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

Arguments:
- `input`: Path to input image (required)
- `checkpoint`: Path to checkpoint file (required)
- `--top_k`: Return top K predictions (default: 5)
- `--category_names`: Path to JSON file mapping categories to names
- `--gpu`: Use GPU for inference if available (flag)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL
- argparse
