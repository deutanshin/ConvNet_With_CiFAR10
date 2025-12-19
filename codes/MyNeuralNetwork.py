import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    # Parameter, which name is dropout_rate, is for testing Overfitting
    def __init__(self, dropout_rate=0):
        super().__init__()

        # define Feature Extractor
        # Implement with Input Size 32x32
        self.feature_extractor = nn.Sequential(

            # 1st Layer
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd Layer
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd Layer
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 4th Layer
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 5th Layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # define Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256*4*4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x