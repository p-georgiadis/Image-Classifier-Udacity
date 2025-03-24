#!/usr/bin/env python3
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models

def get_input_args():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained deep learning model."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument("--category_names", type=str, default=None,
                        help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    return parser.parse_args()

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns a NumPy array.
    """
    original_width, original_height = image.size
    if original_width < original_height:
        new_width = 256
        new_height = int(256 * original_height / original_width)
    else:
        new_height = 256
        new_width = int(256 * original_width / original_height)
    image = image.resize((new_width, new_height))
    
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    arch = checkpoint["architecture"]
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture in checkpoint.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = checkpoint["input_features"]
    hidden_units = checkpoint["hidden_units"]
    output_classes = checkpoint["output_classes"]
    
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_classes),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model

def predict(image_path, model, device, topk=5):
    model.eval()
    
    image = Image.open(image_path)
    np_image = process_image(image)
    image_tensor = torch.from_numpy(np_image).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.exp(output)
    top_probs, top_indices = probabilities.topk(topk)
    
    top_probs = top_probs.cpu().squeeze().tolist()
    top_indices = top_indices.cpu().squeeze().tolist()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    if isinstance(top_indices, int):
        top_indices = [top_indices]
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes

def main():
    args = get_input_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    top_probs, top_classes = predict(args.input, model, device, args.top_k)
    
    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name.get(cls, cls) for cls in top_classes]
    else:
        top_names = top_classes
    
    print("Top probabilities:", top_probs)
    print("Top classes:", top_names)

if __name__ == "__main__":
    main()