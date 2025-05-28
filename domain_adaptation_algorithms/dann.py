import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.autograd import Function

# === Gradient Reversal Layer ===
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def revgrad(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

# === Modules ===
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, bottleneck_dim)
        )

    def forward(self, x):
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self, bottleneck_dim, num_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, alpha=1.0):
        x = revgrad(x, alpha)
        return self.net(x)

class LabelClassifier(nn.Module):
    def __init__(self, bottleneck_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)