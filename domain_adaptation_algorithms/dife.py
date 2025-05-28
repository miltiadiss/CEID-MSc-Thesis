import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === MMD Loss ===
def compute_mmd(x1, x2, kernel='rbf', sigma=1.0):
    def gaussian_kernel(a, b, sigma):
        dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    Kxx = gaussian_kernel(x1, x1, sigma)
    Kyy = gaussian_kernel(x2, x2, sigma)
    Kxy = gaussian_kernel(x1, x2, sigma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

# === Center loss with margin ===
class CenterMarginLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin=1.0, lambda_sep=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_sep = lambda_sep
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, z, labels):
        batch_size = z.size(0)
        dists = (z.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(-1)

        pos = dists[torch.arange(batch_size), labels].mean()
        neg = torch.clamp(self.margin - dists + 1e-6, min=0)
        mask = torch.ones_like(neg).scatter_(1, labels.unsqueeze(1), 0)
        neg = (neg * mask).mean()

        return pos + self.lambda_sep * neg

# === Feature Extractor ===
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Label Classifier ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === DIFE Loss Function ===
def compute_dife_loss(z_s, y_s, z_t, pseudo_y_t, num_classes, classifier, disc_loss_fn, alpha=0.6, beta=0.7):
    cls_loss = F.cross_entropy(classifier(z_s), y_s)

    align_loss = 0.0
    for c in range(num_classes):
        mask_s = (y_s == c)
        mask_t = (pseudo_y_t == c)
        if mask_s.sum() > 1 and mask_t.sum() > 1:
            z_s_c = z_s[mask_s]
            z_t_c = z_t[mask_t]
            align_loss += compute_mmd(z_s_c, z_t_c)
    align_loss /= num_classes

    disc_loss = disc_loss_fn(z_s, y_s)
    total_loss = cls_loss + alpha * align_loss + beta * disc_loss
    return total_loss, cls_loss.item(), align_loss.item(), disc_loss.item()