from torchinfo import summary
from IPython.display import display, Markdown
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def to_onehot(y, num_classes):
    """Converts class and device labels to one-hot encoded tensors.

    Args:
        y (array-like): Class indices.
        num_classes (int): Total number of classes.

    Returns:
        Tensor: One-hot encoded labels.
    """
    return F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=num_classes).float()

def load_data():
    """Loads and preprocesses training and test data.

    Returns:
        dict: Dictionary containing tensors for training/test sets and label counts.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    x = np.load("training_features_mean.npy")
    y = np.load("training_labels.npy")
    d = np.load("training_devices.npy", allow_pickle=True)
    x_test = np.load("test_features_mean.npy")
    y_test = np.load("test_labels.npy")
    d_test = np.load("test_devices.npy", allow_pickle=True)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_test = scaler.transform(x_test)

    le_class = LabelEncoder()
    y_enc = le_class.fit_transform(y)
    y_test_enc = le_class.transform(y_test)
    n_classes = len(np.unique(y_enc))

    le_domain = LabelEncoder()
    d_enc = le_domain.fit_transform(d)
    d_test_enc = le_domain.transform(d_test)
    n_domains = len(np.unique(d_enc))

    return {
        "X_train": torch.tensor(x, dtype=torch.float32),
        "y_train": torch.tensor(y_enc, dtype=torch.long),
        "d_train": to_onehot(d_enc, n_domains),
        "X_test": torch.tensor(x_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test_enc, dtype=torch.long),
        "d_test": to_onehot(d_test_enc, n_domains),
        "n_classes": n_classes,
        "n_domains": n_domains
    }

def build_models(
    model: str,
    input_dim: int,
    n_classes: int,
    n_domains: int,
    bottleneck: int = None,
    latent_dim: int = None,
    FeatureExtractor=None,
    LabelClassifier=None,
    DomainClassifier=None,
    Encoder=None,
    Decoder=None
):
    """Instantiates model components for domain adaptation.

    Args:
        model (str): One of {"dann", "cdan", "davae"}.
        input_dim (int): Input feature dimensionality.
        n_classes (int): Number of target classes.
        n_domains (int): Number of domain labels.
        bottleneck (int, optional): Bottleneck size (for DANN/CDAN).
        latent_dim (int, optional): Latent space size (for DAVAΕ).
        FeatureExtractor (Type[nn.Module], optional): Feature extractor class (for DANN/CDAN).
        LabelClassifier (Type[nn.Module]): Label classifier class.
        DomainClassifier (Type[nn.Module]): Domain classifier class.
        Encoder (Type[nn.Module], optional): Encoder class (for DAVAΕ).
        Decoder (Type[nn.Module], optional): Decoder class (for DAVAΕ).

    Returns:
        dict: Instantiated modules, depending on model type.
    """
    if model == "dann":
        return {
            "G": FeatureExtractor(input_dim, bottleneck),
            "C": LabelClassifier(bottleneck, n_classes),
            "D": DomainClassifier(bottleneck, n_domains),
        }

    elif model == "cdan":
        return {
            "G": FeatureExtractor(input_dim, bottleneck),
            "C": LabelClassifier(bottleneck, n_classes),
            "D": DomainClassifier(bottleneck * n_classes, n_domains),
        }

    else:
        return {
            "Enc": Encoder(input_dim, latent_dim),
            "Dec": Decoder(latent_dim, input_dim),
            "C": LabelClassifier(latent_dim, n_classes),
            "D": DomainClassifier(latent_dim, n_domains),
        }

def plot_losses(logs, model):
    """Plots training and validation loss curves.

    Args:
        logs (dict): Dictionary with loss values over epochs.
        model (str): Domain adaptation variant, either "dann", "cdan" or "davae".
    """
    epochs_range = range(1, len(logs["train_total"]) + 1)

    if model == 'dann' or 'cdan':
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, logs["train_total"], label='Train Total Loss', color='orange')
        plt.plot(epochs_range, logs["val_total"], label='Val Total Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, logs["train_class"], label='Train Classification Loss', color='orange')
        plt.plot(epochs_range, logs["val_class"], label='Val Classification Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classification Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, logs["train_domain"], label='Train Domain Loss', color='orange')
        plt.plot(epochs_range, logs["val_domain"],   label='Val Domain Loss',   color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Domain Loss')
    else:
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 5, 1)
        plt.plot(epochs_range, logs["train_total"], label='Train Total Loss', color='orange')
        plt.plot(epochs_range, logs["val_total"], label='Val Total Loss', color='blue')
        plt.title('Total Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

        plt.subplot(1, 5, 2)
        plt.plot(epochs_range, logs["train_class"], label='Train Class Loss', color='orange')
        plt.plot(epochs_range, logs["val_class"], label='Val Class Loss', color='blue')
        plt.title('Classification Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

        plt.subplot(1, 5, 3)
        plt.plot(epochs_range, logs["train_domain"], label="Train Domain Loss", color="orange")
        plt.plot(epochs_range, logs["val_domain"], label="Val Domain Loss", color="blue")
        plt.title("Domain Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

        plt.subplot(1, 5, 4)
        plt.plot(epochs_range, logs["train_kl"], label="Train KL Loss", color="orange")
        plt.plot(epochs_range, logs["val_kl"], label="Val KL Loss", color="blue")
        plt.title("KL Divergence"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

        plt.subplot(1, 5, 5)
        plt.plot(epochs_range, logs["train_recon"], label="Train Reconstruction Loss", color="orange")
        plt.plot(epochs_range, logs["val_recon"], label="Val Reconstruction Loss", color="blue")
        plt.title("Reconstruction Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout()
    plt.show()

def plot_2stage_losses(vae_logs, dann_logs, vae_epochs):
    """Plots training and validation losses across two-stage training (VAE → DANN).

    Combines the pretraining losses from the VAE stage (reconstruction, KL divergence)
    with the supervised + domain adaptation losses from the DANN stage (classification, domain).
    Adds a vertical dashed line to indicate the transition between the two stages.

    Args:
        vae_logs (dict): Dictionary of VAE training logs with keys:
            - "train_recon" (list[float]): Reconstruction losses on training set.
            - "val_recon"   (list[float]): Reconstruction losses on validation set.
            - "train_kl"    (list[float]): KL divergence losses on training set.
            - "val_kl"      (list[float]): KL divergence losses on validation set.
        dann_logs (dict): Dictionary of DANN training logs with keys:
            - "train_class"  (list[float]): Classification losses on training set.
            - "val_class"    (list[float]): Classification losses on validation set.
            - "train_domain" (list[float]): Domain losses on training set.
            - "val_domain"   (list[float]): Domain losses on validation set.
        vae_epochs (int): Number of epochs used in the VAE pretraining stage.

    Returns:
        None: Displays a matplotlib figure with four subplots:
            (1) Reconstruction loss,
            (2) KL divergence,
            (3) Classification loss,
            (4) Domain loss.
    """
    
    total_epochs = vae_epochs + len(dann_logs["train_class"])
    epochs = np.arange(1, total_epochs + 1)

    train_recon = vae_logs["train_recon"] + [np.nan] * len(dann_logs["train_class"])
    val_recon   = vae_logs["val_recon"]   + [np.nan] * len(dann_logs["val_class"])
    train_kl    = vae_logs["train_kl"]    + [np.nan] * len(dann_logs["train_class"])
    val_kl      = vae_logs["val_kl"]      + [np.nan] * len(dann_logs["val_class"])

    train_class = [np.nan] * vae_epochs + dann_logs["train_class"]
    val_class   = [np.nan] * vae_epochs + dann_logs["val_class"]
    train_dom   = [np.nan] * vae_epochs + dann_logs["train_domain"]
    val_dom     = [np.nan] * vae_epochs + dann_logs["val_domain"]

    plt.figure(figsize=(24, 6))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_recon, label="Train Recon", color="orange")
    plt.plot(epochs, val_recon, label="Val Recon", color="blue")
    plt.axvline(vae_epochs, color="black", linestyle="--", label="→ DANN Start")
    plt.title("Reconstruction Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_kl, label="Train KL", color="orange")
    plt.plot(epochs, val_kl, label="Val KL", color="blue")
    plt.axvline(vae_epochs, color="black", linestyle="--")
    plt.title("KL Divergence"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_class, label="Train Class", color="orange")
    plt.plot(epochs, val_class, label="Val Class", color="blue")
    plt.axvline(vae_epochs, color="black", linestyle="--")
    plt.title("Classification Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_dom, label="Train Domain", color="orange")
    plt.plot(epochs, val_dom, label="Val Domain", color="blue")
    plt.axvline(vae_epochs, color="black", linestyle="--")
    plt.title("Domain Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.suptitle("Losses Across Two-Stage Training (VAE → DANN)", fontsize=16)
    plt.tight_layout(); plt.show()

def print_model_summary(title, model, input_size):
    """
    Displays a PyTorch model summary in a Jupyter notebook with a markdown header.

    Args:
        title (str): The title to display above the model summary.
        model (nn.Module): The PyTorch model to summarize.
        input_size (tuple): Input size in the form (batch_size, input_dim).

    Returns:
        None
    """
    display(Markdown(f"### {title}"))
    print(summary(model, input_size=input_size))

def plot_losses(logs):
    epochs_range = range(1, len(logs["train_total"]) + 1)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.plot(epochs_range, logs["train_total"], label='Train Total Loss', color='orange')
    plt.plot(epochs_range, logs["val_total"], label='Val Total Loss', color='blue')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 5, 2)
    plt.plot(epochs_range, logs["train_class"], label='Train Class Loss', color='orange')
    plt.plot(epochs_range, logs["val_class"], label='Val Class Loss', color='blue')
    plt.title('Classification Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 5, 3)
    plt.plot(epochs_range, logs["train_domain"], label="Train Domain Loss", color="orange")
    plt.plot(epochs_range, logs["val_domain"], label="Val Domain Loss", color="blue")
    plt.title("Domain Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 5, 4)
    plt.plot(epochs_range, logs["train_kl"], label="Train KL Loss", color="orange")
    plt.plot(epochs_range, logs["val_kl"], label="Val KL Loss", color="blue")
    plt.title("KL Divergence"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 5, 5)
    plt.plot(epochs_range, logs["train_recon"], label="Train Reconstruction Loss", color="orange")
    plt.plot(epochs_range, logs["val_recon"], label="Val Reconstruction Loss", color="blue")
    plt.title("Reconstruction Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.tight_layout(); plt.show()