import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x

def my_hook(module, input, output):
    print('Output of conv2:', output.shape)

model = MyModel()
conv2 = model.conv2

# Register the hook
conv2.register_forward_hook(my_hook)

# Test the model
x = torch.randn(1, 3, 32, 32)
output = model(x)
