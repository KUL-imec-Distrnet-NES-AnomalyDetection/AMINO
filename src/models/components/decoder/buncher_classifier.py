import torch.nn as nn


class BuncherClassisfier(nn.Module):
    def __init__(self, buncher, classifier):
        super().__init__()
        self.buncher = buncher
        self.classifier = classifier

    def forward(self, x):
        h = self.buncher(x)
        y = self.classifier(h)
        return {
            "classify": y,
            "bunch": h,
        }


class ClassifierBuncher(BuncherClassisfier):
    def forward(self, x):
        y1 = self.classifier(x)
        y2 = self.buncher(y1)
        return {
            "classify": y1,
            "bunch": y2,
        }
