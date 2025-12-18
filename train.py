import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from MyNeuralNetwork import MyNeuralNetwork
from dataset import get_dataloaders

# Weight Initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # He Init, Mean 0, random Std
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # bias init to 0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training config: Overfit={args.overfit}, Epochs={args.epochs}, Batch={args.batch}, LR={args.lr}")
    
    # from dataset.py
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch, 
        overfit=args.overfit
    )

    # Set model mode
    if args.overfit:
        dropout_p = 0.0
        wd = 0.0
    else:
        # None Overfit mode, apply 2 methods of Regularization
        dropout_p = 0.5 # Dropout
        wd = 1e-4       # Weight Decay
    model = MyNeuralNetwork(dropout_rate=dropout_p).to(device)
    model.apply(init_weights)

    # Use Cross Entrophy Loss Function and Adam Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print(f"Starting Train (Device: {device}, Mode: {args.overfit})")
    
    # default epoch is 40
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                  f"Val Acc: {epoch_val_acc:.2f}%")
            
    
    if not args.overfit:
        print(f"=========================================")
        print(f"Final Test is starting")
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        final_acc = 100. * test_correct / test_total
        print(f"Final Test Accuracy: {final_acc:.2f}%")

        print(f"=========================================")


    # Visualization for Report
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Change')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Change')
    plt.legend()

    
    # Save result plot and model parameters by model for analysis report
    mode_name = "overfit" if args.overfit else "regularized"

    plt.savefig(f'result_{mode_name}.png')
    print(f"Training finished. Saved result plot as result_{mode_name}.png")

    torch.save(model.state_dict(), f'model_{mode_name}.pth')
    print(f"Saved model parameters to model_{mode_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', action='store_true', help='Run in overfit mode')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    main(args)