import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from domain_adaptation_algorithms.dann import *

# === Modules ===
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, z):
        return self.fc(z)

# Label Classifier
class LabelClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, z):
        return self.fc(z)

# Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, latent_dim, num_domains):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_domains)
        )

    def forward(self, z, alpha):
        z = revgrad(z, alpha)
        return self.fc(z)

# === VAE Loss ===
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl

# === Training ===
def train_davae(encoder, decoder, label_clf, domain_clf, data, beta, class_weight, gamma, epochs):
    X_train = data["X_train"]
    y_train = data["y_train"]
    d_train = data["d_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    d_test = data["d_test"]

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(label_clf.parameters()) +
        list(domain_clf.parameters()),
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    train_total_losses, train_class_losses, train_domain_losses, train_kl_losses = [], [], [], []
    val_total_losses, val_class_losses, val_domain_losses, val_kl_losses = [], [], [], []
    train_recon_losses, val_recon_losses = [], []

    min_loss = float("inf")

    for epoch in range(epochs):
        encoder.train(); decoder.train(); label_clf.train(); domain_clf.train()
        p = epoch / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        z, mu, logvar = encoder(X_train)
        x_recon = decoder(z)
        logits = label_clf(z)
        domain_logits = domain_clf(z, alpha)

        recon, kl = vae_loss(x_recon, X_train, mu, logvar)
        loss_c = loss_class(logits, y_train)
        loss_d = loss_domain(domain_logits, d_train)
        total_loss = recon + beta * kl + class_weight * loss_c + gamma * loss_d

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_total_losses.append(total_loss.item())
        train_class_losses.append(loss_c.item())
        train_domain_losses.append(loss_d.item())
        train_kl_losses.append(kl.item())
        train_recon_losses.append(recon.item())

        scheduler.step()

        # Validation
        encoder.eval(); decoder.eval(); label_clf.eval(); domain_clf.eval()
        with torch.no_grad():
            z_val, mu_val, logvar_val = encoder(X_test)
            x_recon_val = decoder(z_val)
            logits_val = label_clf(z_val)
            domain_logits_val = domain_clf(z_val, alpha)

            recon_val, kl_val = vae_loss(x_recon_val, X_test, mu_val, logvar_val)
            loss_c_val = loss_class(logits_val, y_test)
            loss_d_val = loss_domain(domain_logits_val, d_test)
            val_total = recon_val + beta * kl_val + class_weight * loss_c_val + gamma * loss_d_val

        val_total_losses.append(val_total.item())
        val_class_losses.append(loss_c_val.item())
        val_domain_losses.append(loss_d_val.item())
        val_kl_losses.append(kl_val.item())
        val_recon_losses.append(recon_val.item())

        if epoch == 0 or total_loss.item() < min_loss:
            min_loss = total_loss.item()
            torch.save(encoder.state_dict(), "best_encoder.pth")
            torch.save(label_clf.state_dict(), "best_classifier.pth")

    encoder.eval()
    with torch.no_grad():
        emb_train = encoder(X_train)[0].cpu().numpy()
        emb_test = encoder(X_test)[0].cpu().numpy()

    return {
        "train_total": train_total_losses,
        "train_class": train_class_losses,
        "train_domain": train_domain_losses,
        "train_kl": train_kl_losses,
        "train_recon": train_recon_losses,   
        "val_total": val_total_losses,
        "val_class": val_class_losses,
        "val_domain": val_domain_losses,
        "val_kl": val_kl_losses,
        "val_recon": val_recon_losses        
    }, emb_train, emb_test

def train_vae_stage(encoder, decoder, data, beta, epochs):
    """Pretrains a Variational Autoencoder (VAE) for reconstruction.

    This stage trains the encoder–decoder pair using reconstruction and KL divergence losses,
    without classification or domain adaptation.

    Args:
        encoder (nn.Module): VAE encoder network. Must return (z, mu, logvar).
        decoder (nn.Module): VAE decoder network. Reconstructs input from latent representation.
        data (dict): Dictionary with training and test sets. Must contain:
            - "X_train" (Tensor): Training features.
            - "X_test"  (Tensor): Test features.
        beta (float): Weight for KL divergence term.
        epochs (int): Number of training epochs.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]:
            - logs (dict): Training and validation loss curves with keys:
                {"train_recon", "train_kl", "val_recon", "val_kl"}.
            - emb_train (ndarray): Latent embeddings of training set after training.
            - emb_test (ndarray): Latent embeddings of test set after training.
    """
    
    X_train = data["X_train"]
    X_test = data["X_test"]

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_recon_losses, train_kl_losses = [], []
    val_recon_losses, val_kl_losses = [], []

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        z, mu, logvar = encoder(X_train)
        x_recon = decoder(z)
        recon, kl = vae_loss(x_recon, X_train, mu, logvar)
        loss = recon + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_recon_losses.append(recon.item())
        train_kl_losses.append(kl.item())

        # Validation
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            z_val, mu_val, logvar_val = encoder(X_test)
            x_recon_val = decoder(z_val)
            recon_val, kl_val = vae_loss(x_recon_val, X_test, mu_val, logvar_val)

        val_recon_losses.append(recon_val.item())
        val_kl_losses.append(kl_val.item())

    return {
        "train_recon": train_recon_losses,
        "train_kl": train_kl_losses,
        "val_recon": val_recon_losses,
        "val_kl": val_kl_losses
    }

def train_dann_stage(encoder, label_clf, domain_clf, data, class_weight, gamma, epochs, start_epoch=0):
    """Trains the second stage: supervised classification + domain adaptation (DANN).

    This stage freezes the VAE decoder and continues training the encoder jointly with
    a label classifier and a domain classifier, using gradient reversal for adversarial adaptation.

    Args:
        encoder (nn.Module): Encoder network returning (z, mu, logvar).
        label_clf (nn.Module): Label classifier head.
        domain_clf (nn.Module): Domain classifier head (with GRL inside).
        data (dict): Dictionary with training and test sets. Must contain:
            - "X_train" (Tensor): Training features.
            - "y_train" (Tensor): Training class labels.
            - "d_train" (Tensor): Training domain labels (one-hot or indices).
            - "X_test"  (Tensor): Test features.
            - "y_test"  (Tensor): Test class labels.
            - "d_test"  (Tensor): Test domain labels.
        class_weight (float): Weighting factor for classification loss.
        gamma (float): Weighting factor for domain loss.
        epochs (int): Number of training epochs for this stage.
        start_epoch (int, optional): Offset for epoch counter.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]:
            - logs (dict): Training and validation loss curves with keys:
                {"train_class", "train_domain", "val_class", "val_domain"}.
            - emb_train (ndarray): Latent embeddings of training set after training.
            - emb_test (ndarray): Latent embeddings of test set after training.
    """

    X_train = data["X_train"]
    y_train = data["y_train"]
    d_train = data["d_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    d_test  = data["d_test"]

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(label_clf.parameters()) + list(domain_clf.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    train_class_losses, train_domain_losses = [], []
    val_class_losses, val_domain_losses = [], []

    for epoch in range(start_epoch, start_epoch + epochs):
        encoder.train(); label_clf.train(); domain_clf.train()
        p = (epoch - start_epoch) / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        z = encoder(X_train)[0]
        class_logits = label_clf(z)
        domain_logits = domain_clf(z, alpha)

        loss_c = loss_class(class_logits, y_train)
        loss_d = loss_domain(domain_logits, d_train)
        total_loss = class_weight * loss_c + gamma * loss_d

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        train_class_losses.append(loss_c.item())
        train_domain_losses.append(loss_d.item())

        # Validation
        encoder.eval(); label_clf.eval(); domain_clf.eval()
        with torch.no_grad():
            z_val = encoder(X_test)[0]
            class_logits_val = label_clf(z_val)
            domain_logits_val = domain_clf(z_val, alpha)

            loss_c_val = loss_class(class_logits_val, y_test)
            loss_d_val = loss_domain(domain_logits_val, d_test)

        val_class_losses.append(loss_c_val.item())
        val_domain_losses.append(loss_d_val.item())

    encoder.eval()
    with torch.no_grad():
        emb_train = encoder(X_train)[0].cpu().numpy()
        emb_test = encoder(X_test)[0].cpu().numpy()

    return {
        "train_class": train_class_losses,
        "train_domain": train_domain_losses,
        "val_class": val_class_losses,
        "val_domain": val_domain_losses
    }, emb_train, emb_test