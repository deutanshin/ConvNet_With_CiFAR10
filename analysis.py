import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from MyNeuralNetwork import MyNeuralNetwork
from dataset import get_dataloaders
import argparse

# CIFAR-10 class name
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def unnormalize(img):
    # regularized tensor to image
    # same mean, std with dataset.py
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = img * std + mean
    return img

def imshow(img, title, ax):
    # print image to matplot lib
    
    img = unnormalize(img.cpu())
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0)) 
    npimg = np.clip(npimg, 0, 1)
    
    ax.imshow(npimg)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def run_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # test dataset load
    _, _, test_loader = get_dataloaders(batch_size=1, overfit=False)
    
    # model, weight load
    # Need pth file to run analysis.py

    model = MyNeuralNetwork(dropout_rate=0.5).to(device)
    
    model_path = f"model_{args.mode}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} 파일을 찾을 수 없습니다. 학습을 먼저 실행하세요.")
        return

    model.eval()

    # collect wrong or correct cases
    correct_samples = []
    wrong_samples = []
    
    print("Collecting samples...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            label_val = labels.item()
            pred_val = predicted.item()

            # save (image, predict, label)
            if pred_val == label_val and len(correct_samples) < 5:
                correct_samples.append((images[0], pred_val, label_val))
            elif pred_val != label_val and len(wrong_samples) < 5:
                wrong_samples.append((images[0], pred_val, label_val))
            
            # until 5 cases each one
            if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                break
    
    # save
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # correct case
    for i in range(5):
        img, pred, label = correct_samples[i]
        title = f"[Correct]\nTrue: {classes[label]}\nPred: {classes[pred]}"
        imshow(img, title, axes[0][i])
        
    # wrongcase
    for i in range(5):
        img, pred, label = wrong_samples[i]
        title = f"[Wrong]\nTrue: {classes[label]}\nPred: {classes[pred]}"
        imshow(img, title, axes[1][i])

    plt.tight_layout()
    save_filename = f"analysis_{args.mode}.png"
    plt.savefig(save_filename)
    print(f"Analysis saved to {save_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parmeters --mode (overfit or regularized)
    parser.add_argument('--mode', type=str, required=True, choices=['overfit', 'regularized'],
                        help="Choose model to analyze: 'overfit' or 'regularized'")
    args = parser.parse_args()
    run_analysis(args)