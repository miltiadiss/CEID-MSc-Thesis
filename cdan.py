import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from domain_adaptation_algorithms.dann import *

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# === Modules ===
# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# Label Classifier
class LabelClassifier(nn.Module):
    def __init__(self, bottleneck, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, alpha):
        x = revgrad(x, alpha)
        return self.net(x)

# === Conditional Feature Map ===
def conditional_map(features, y_soft):
    return torch.bmm(y_soft.unsqueeze(2), features.unsqueeze(1)).view(features.size(0), -1)

# === Training ===
def train_cdan(G, C, D, data, lambda_domain=0.2, epochs=200):
    X_train = data["X_train"]
    y_train = data["y_train"]
    d_train = data["d_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    d_test = data["d_test"]
    n_classes = data["n_classes"]

    optimizer = torch.optim.Adam(list(G.parameters()) + list(C.parameters()) + list(D.parameters()), lr=1e-3)
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.MSELoss()

    train_total_losses, train_class_losses, train_domain_losses = [], [], []
    val_total_losses, val_class_losses, val_domain_losses = [], [], []
    min_loss = float("inf")

    for epoch in range(epochs):
        G.train(); C.train(); D.train()
        p = epoch / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        feat = G(X_train)
        logits = C(feat)
        y_soft = F.softmax(logits, dim=1)
        cond_feat = conditional_map(feat, y_soft)
        domain_logits = D(cond_feat, alpha=alpha)

        loss_c = loss_class(logits, y_train)
        loss_d = loss_domain(F.softmax(domain_logits, dim=1), d_train)
        total_loss = loss_c - lambda_domain * loss_d

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_total_losses.append(total_loss.item())
        train_class_losses.append(loss_c.item())
        train_domain_losses.append(loss_d.item())

        # Validation
        G.eval(); C.eval(); D.eval()
        with torch.no_grad():
            feat_val = G(X_test)
            logits_val = C(feat_val)
            y_val_soft = F.softmax(logits_val, dim=1)
            cond_feat_val = conditional_map(feat_val, y_val_soft)
            domain_logits_val = D(cond_feat_val, alpha=alpha)

            loss_c_val = loss_class(logits_val, y_test)
            loss_d_val = loss_domain(F.softmax(domain_logits_val, dim=1), d_test)
            val_total = loss_c_val - lambda_domain * loss_d_val

        val_total_losses.append(val_total.item())
        val_class_losses.append(loss_c_val.item())
        val_domain_losses.append(loss_d_val.item())

        if epoch == 0 or total_loss.item() < min_loss:
            min_loss = total_loss.item()
            torch.save(G.state_dict(), "best_G.pth")
            torch.save(C.state_dict(), "best_C.pth")

    G.eval()
    with torch.no_grad():
        emb_train = G(X_train).cpu().numpy()
        emb_test = G(X_test).cpu().numpy()

    return {
        "train_total": train_total_losses,
        "train_class": train_class_losses,
        "train_domain": train_domain_losses,
        "val_total": val_total_losses,
        "val_class": val_class_losses,
        "val_domain": val_domain_losses
    }, emb_train, emb_test

