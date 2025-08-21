import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# === Modules ===
# Gradient Reversal Layer (GRL)
class GradientReversal(torch.autograd.Function):
    """Implements the gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Args:
            x (Tensor): Input tensor.
            alpha (float): Reversal coefficient.

        Returns:
            Tensor: Unmodified input.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output (Tensor): Gradient of the loss.

        Returns:
            Tuple[Tensor, None]: Reversed gradient.
        """
        return -ctx.alpha * grad_output, None

def revgrad(x, alpha):
    """Applies the gradient reversal function."""
    return GradientReversal.apply(x, alpha)

# Feature Extractor
class FeatureExtractor(nn.Module):
    """Feature extractor network for DANN."""
    def __init__(self, input_dim, bottleneck=128):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            bottleneck (int): Output dimensionality of feature space.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Extracted features.
        """
        return self.model(x)

# Label Classifier
class LabelClassifier(nn.Module):
    """Label classifier network for DANN."""
    def __init__(self, bottleneck, num_classes):
        """
        Args:
            bottleneck (int): Input dimensionality from feature extractor.
            num_classes (int): Number of target classes.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features.

        Returns:
            Tensor: Class logits.
        """
        return self.model(x)

# Domain Classifier
class DomainClassifier(nn.Module):
    """Domain classifier network for DANN."""
    def __init__(self, bottleneck, num_domains):
        """
        Args:
            bottleneck (int): Input dimensionality from feature extractor.
            num_domains (int): Number of domains/devices.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, alpha):
        """
        Args:
            x (Tensor): Input features.
            alpha (float): GRL coefficient.

        Returns:
            Tensor: Domain logits.
        """
        x = revgrad(x, alpha)
        return self.model(x)
    
# Wrapper for Domain Classifier summary
class DomainClassifierWrapper(nn.Module):
    def __init__(self, domain_classifier, alpha=1.0):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.alpha = alpha

    def forward(self, x):
        return self.domain_classifier(x, self.alpha)

# === Training ===
def train_dann(G, C, D, data, lambda_domain=0.1, epochs=200):
    """Trains the DANN model and saves embeddings.

    Args:
        G (nn.Module): Feature extractor.
        C (nn.Module): Label classifier.
        D (nn.Module): Domain classifier.
        data (dict): Preprocessed training and test data.
        lambda_domain (float): Domain loss coefficient.
        epochs (int): Number of training epochs.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]: Loss logs, train embeddings, test embeddings.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    d_train = data["d_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    d_test = data["d_test"]

    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.MSELoss()
    optimizer = torch.optim.Adam(list(G.parameters()) + list(C.parameters()) + list(D.parameters()), lr=1e-3)

    train_total_losses, train_class_losses, train_domain_losses = [], [], []
    val_total_losses, val_class_losses, val_domain_losses = [], [], []
    min_loss = float("inf")

    for epoch in range(epochs):
        G.train(); C.train(); D.train()
        p = epoch / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        feats = G(X_train)
        class_logits = C(feats)
        domain_logits = D(feats, alpha=alpha)

        loss_c = loss_class(class_logits, y_train)
        loss_d = loss_domain(F.softmax(domain_logits, dim=1), d_train)
        total_loss = loss_c - lambda_domain * loss_d

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_total_losses.append(total_loss.item())
        train_class_losses.append(loss_c.item())
        train_domain_losses.append(loss_d.item())

        G.eval(); C.eval(); D.eval()
        with torch.no_grad():
            feats_val = G(X_test)
            class_logits_val = C(feats_val)
            domain_logits_val = D(feats_val, alpha=alpha)

            val_loss_c = loss_class(class_logits_val, y_test)
            val_loss_d = loss_domain(F.softmax(domain_logits_val, dim=1), d_test)
            val_total = val_loss_c - lambda_domain * val_loss_d

        val_total_losses.append(val_total.item())
        val_class_losses.append(val_loss_c.item())
        val_domain_losses.append(val_loss_d.item())

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
