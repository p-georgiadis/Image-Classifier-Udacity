#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser(
        description="Train a deep neural network for flower classification."
    )
    parser.add_argument("data_dir", help="Directory of the dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", 
                        help="File path to save checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", 
                        help='Pretrained model architecture: "vgg16" or "vgg13"')
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=512, 
                        help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", 
                        help="Use GPU for training if available")
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir  = os.path.join(data_dir, "test")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    dataloaders = {
        key: torch.utils.data.DataLoader(image_datasets[key], batch_size=32, shuffle=(key=='train'))
        for key in image_datasets
    }
    
    return image_datasets, dataloaders

def build_model(arch, hidden_units, num_classes):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Choose 'vgg16' or 'vgg13'.")
    
    # Freeze pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    return model

def train_model(model, dataloaders, device, epochs, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    steps = 0
    print_every = 40

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs_val, labels_val in dataloaders['valid']:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        outputs_val = model(inputs_val)
                        loss_val = criterion(outputs_val, labels_val)
                        valid_loss += loss_val.item()
                        ps = torch.exp(outputs_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_val.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer

def save_checkpoint(model, image_datasets, arch, hidden_units, epochs, optimizer, save_path):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'architecture': arch,
        'input_features': model.classifier[0].in_features,
        'hidden_units': hidden_units,
        'output_classes': len(image_datasets['train'].classes),
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    args = get_input_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    image_datasets, dataloaders = load_data(args.data_dir)
    num_classes = len(image_datasets['train'].classes)
    
    model = build_model(args.arch, args.hidden_units, num_classes)
    
    model, optimizer = train_model(model, dataloaders, device, args.epochs, args.learning_rate)
    
    save_checkpoint(model, image_datasets, args.arch, args.hidden_units, args.epochs, optimizer, args.save_dir)

if __name__ == "__main__":
    main()