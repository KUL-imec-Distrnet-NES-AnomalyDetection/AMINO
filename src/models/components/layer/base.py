import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self, lambda_func):
        super().__init__()
        if type(lambda_func) is str:
            lambda_func = eval(lambda_func)
        self.lambda_func = lambda_func
    
    def forward(self, x):
        return self.lambda_func(x)
