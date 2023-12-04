import torch
import torch.nn as nn

class BuncherClassisfier(nn.Module):
    def __init__(self, buncher, classifier):
        super().__init__()
        self.buncher = buncher
        self.classifier = classifier
    
    def forward(self, x):
        h = self.buncher(x)
        y = self.classifier(h)
        return y

class ClassifierBuncher(BuncherClassisfier):
    def forward(self, x):
        y1 = self.classifier(x)
        y2 = self.buncher(y1)
        return y2