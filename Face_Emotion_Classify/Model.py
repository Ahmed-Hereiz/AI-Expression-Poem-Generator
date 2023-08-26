import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10)
        )
        
        # Freeze the features layers
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x