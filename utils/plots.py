import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import librosa
import librosa.display
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Union
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from utils.audio_preprocessing import classify
from typing import Dict
import scipy.signal

# == Functions ==
def plot_dataset_waveforms(audio_folder, k, sr):
    """
    Plots the first k waveforms with colored waveform segments based on respiratory cycle annotations.

    Args:
        audio_folder (str): Folder with .wav and .txt files.
        k (int): Number of waveforms to plot.
        sr (int): Sampling rate.
    """

    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e",
        "Both": "#219ebc"
    }

    wav_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])[:k]

    for fname in wav_files:
        wav_path = os.path.join(audio_folder, fname)
        txt_path = wav_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            continue

        y, _ = librosa.load(wav_path, sr=sr)
        t = np.arange(len(y)) / sr
        annotations = np.loadtxt(txt_path)  # start, end, crackle, wheeze

        plt.figure(figsize=(12, 3))
        plt.title(f"{fname}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        for row in annotations:
            start, end, crackle, wheeze = row
            label = classify(int(crackle), int(wheeze))
            color = label_colors[label]

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Plot waveform segment with different color for each class
            t_seg = t[start_sample:end_sample]
            y_seg = y[start_sample:end_sample]
            plt.plot(t_seg, y_seg, color=color, linewidth=1.2)

            plt.axvline(start, color="black", linestyle="--", linewidth=0.8)
            plt.axvline(end, color="black", linestyle="--", linewidth=0.8)

        handles = [plt.Line2D([0], [0], color=c, lw=3) for c in label_colors.values()]
        plt.legend(handles, label_colors.keys(), loc="upper right")
        plt.tight_layout()
        plt.show()

def plot_feature_vs_class_heatmap(features: np.ndarray, labels: np.ndarray, feature_names: list = None):
    """Plot a heatmap with features as rows and classes as columns.

    Each cell shows the mean value of the feature for the given class.

    Args:
        features (np.ndarray): Array of shape (samples, features) containing the feature values.
        labels (np.ndarray): Array of shape (samples,) containing the class labels.
        feature_names (list, optional): List of feature names. If None, generic names will be used.

    Returns:
        None: The function displays the heatmap plot.
    """
    unique_classes = np.unique(labels)
    heatmap_data = []

    for cls in unique_classes:
        cls_feats = features[labels == cls]
        mean_feats = cls_feats.mean(axis=0)
        heatmap_data.append(mean_feats)

    heatmap_data = np.array(heatmap_data).T  # shape: (features, classes)

    plt.figure(figsize=(10, max(6, features.shape[1] * 0.3)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=unique_classes, yticklabels=feature_names)
    plt.xlabel("Class")
    plt.ylabel("Feature")
    plt.title("Mean feature values per class")
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_heatmap(features: np.ndarray, feature_names: list = None):
    """Plot a lower triangle correlation heatmap between features.

    Args:
        features (np.ndarray): Array of shape (samples, features) with feature values.
        feature_names (list, optional): List of feature names. If None, generic names are used.

    Returns:
        None: Displays the heatmap plot.
    """
    # Compute correlation matrix
    corr = np.corrcoef(features, rowvar=False)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                cbar_kws={"label": "Correlation"})
    plt.title("Correlation Heatmap Between Features")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list
) -> None:
    """Plot KDE and boxplot grids of all features grouped by class.

    Args:
        features (np.ndarray): Array of shape (cycles, features).
        labels (np.ndarray): Array of shape (cycles,) with class labels.
        feature_names (list): List of feature names (length = features.shape[1]).

    Returns:
        None: Displays the grid plots.
    """
    num_cycles, num_features = features.shape
    assert len(feature_names) == num_features, "Number of feature names must match features.shape[1]"

    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }
    unique_labels = np.unique(labels)

    nrows, ncols = 11, 4

    # KDE Plots
    fig_kde, axes_kde = plt.subplots(nrows, ncols, figsize=(20, 30))
    axes_kde = axes_kde.flatten()

    for i in range(len(axes_kde)):
        ax = axes_kde[i]
        if i < num_features:
            for label in unique_labels:
                color = class_colors.get(label, "#333333")
                sns.kdeplot(
                    features[labels == label, i],
                    label=str(label) if i == 0 else None, 
                    fill=True,
                    color=color,
                    ax=ax
                )
            ax.set_title(feature_names[i])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.axis("off")

    handles, legend_labels = axes_kde[0].get_legend_handles_labels()
    fig_kde.legend(handles, legend_labels, title="Class", loc="upper center", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Boxplots
    fig_box, axes_box = plt.subplots(nrows, ncols, figsize=(20, 30))
    axes_box = axes_box.flatten()

    for i in range(len(axes_box)):
        ax = axes_box[i]
        if i < num_features:
            data = []
            group_labels = []
            for label in unique_labels:
                values = features[labels == label, i]
                data.append(values)
                group_labels.extend([str(label)] * len(values))
            data_flat = np.concatenate(data)
            palette = [class_colors.get(lbl, "#333333") for lbl in unique_labels]
            sns.boxplot(x=group_labels, y=data_flat, palette=palette, ax=ax)
            ax.set_title(feature_names[i])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_anova_stats(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list
) -> None:
    """Compute ANOVA F-score and p-value for each feature and visualize them.

    Args:
        features (np.ndarray): Array of shape (cycles, features).
        labels (np.ndarray): Array of shape (cycles,) with class labels.
        feature_names (list): List of feature names (length = features.shape[1]).

    Returns:
        None: Displays barplots of F-scores and p-values.
    """
    num_features = features.shape[1]
    assert len(feature_names) == num_features, "Number of feature names must match features.shape[1]"

    unique_labels = np.unique(labels)

    f_scores = []
    p_values = []

    for i in range(num_features):
        groups = [features[labels == label, i] for label in unique_labels]
        f_val, p_val = f_oneway(*groups)
        f_scores.append(f_val)
        p_values.append(p_val)

    f_scores = np.array(f_scores)
    p_values = np.array(p_values)

    # Plot F-score
    plt.figure(figsize=(16, 6))
    sns.barplot(x=feature_names, y=f_scores, color="blue")
    plt.xticks(rotation=90)
    plt.title("ANOVA F-scores per feature")
    plt.ylabel("F-score")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()

    # Plot p-values (log scale)
    plt.figure(figsize=(16, 6))
    sns.barplot(x=feature_names, y=p_values, color="orange")
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.title("ANOVA p-values per feature (log scale)")
    plt.ylabel("p-value (log scale)")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_mean_feature_evolution_framewise(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list
) -> None:
    """Plot mean evolution over frames of all features grouped by class.

    Args:
        features (np.ndarray): Array of shape (cycles, frames, features).
        labels (np.ndarray): Array of class labels of shape (cycles,).
        feature_names (list): List of feature names (length = features.shape[2]).

    Returns:
        None: Displays the grid plot.
    """
    n_cycles, n_frames, n_features = features.shape

    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }

    nrows, ncols = 11, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 30))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_features:
            for label in np.unique(labels):
                mask = labels == label
                if np.sum(mask) == 0:
                    continue
                mean_curve = features[mask, :, i].mean(axis=0)
                color = label_colors.get(label, "gray")
                ax.plot(mean_curve, label=label if i == 0 else None, color=color)

            ax.set_title(feature_names[i])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Class", loc="upper center", ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_mean_mfcc_heatmap(
    features: np.ndarray,
    n_mfcc: int
) -> None:
    """
    Plot a single heatmap for all the MFCCs where each cell is the mean MFCC value at that frame.

    Args:
        features (np.ndarray): Array of shape (n_cycles, n_frames, n_features).
                               Assumes MFCCs are the first n_mfcc columns.
        n_mfcc (int): Number of MFCCs at the start of the feature dimension.

    Returns:
        None
    """
    n_cycles, n_frames, n_features = features.shape
    assert n_mfcc <= n_features, "n_mfcc exceeds available feature columns."

    X = features
    # Keep only the MFCCs and take the mean over frames
    mean_mfcc = X[:, :, :13].mean(axis=0)  # (n_frames, n_mfcc)

    plt.figure(figsize=(10, 4))
    im = plt.imshow(mean_mfcc.T, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Frame")
    plt.ylabel("MFCC index")
    plt.title("Mean MFCC frame-wise evolution")
    cbar = plt.colorbar(im)
    cbar.set_label("Mean MFCC value")
    plt.tight_layout()
    plt.show()

def plot_mean_logmel_heatmap(
    features: np.ndarray,
    n_mfcc: int,
    n_mels: int
) -> None:
    """
    Plot a single heatmap for all the log-Mel filterbanks where each cell is the
    mean value at that frame.

    Args:
        features (np.ndarray): Array of shape (n_cycles, n_frames, n_features).
                               Assumes log-Mel features come after MFCC and spectral features.
        n_mfcc (int): Number of MFCCs at the start of the feature dimension.
        n_mels (int): Number of log-Mel bands.

    Returns:
        None
    """
    n_cycles, n_frames, n_features = features.shape

    # Keep only the log-mel filterbanks
    X = features[:, :, 20: 40]

    # Mean over cycles
    mean_logmel = X.mean(axis=0)  # (n_frames, n_mels)

    # Heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mean_logmel.T, aspect="auto", origin="lower", cmap="magma")
    plt.xlabel("Frame")
    plt.ylabel("Log-Mel index")
    plt.title("Mean Log-Mel frame-wise evolution")
    cbar = plt.colorbar(im)
    cbar.set_label("Mean Log-Mel value")
    plt.tight_layout()
    plt.show()

def plot_all_features_on_spectrogram_tensor(
    fname: str,
    features: np.ndarray,
    cycle_idx: int,
    label: str,
    sampling_rate: int,
    frame_length: float,
    hop_length: int,
    n_mels: int,
    feature_names: list
) -> None:
    """
    Plot spectrogram for a cycle and overlay all features in a grid (11x4).

    Args:
        fname (str): Audio file name.
        features (np.ndarray): Array (cycles, frames, features).
        cycle_idx (int): Index of the cycle.
        label (str): Class label.
        sampling_rate (int): Sampling rate.
        frame_length (float): Frame length in seconds.
        hop_length (int): Hop length in samples.
        n_mels (int): Number of Mel bands.
        feature_names (list): List of feature names.
    """
    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e",
        "Both": "#219ebc"
    }

    cleaned_folder = "cycles/cleaned_audio_cycles"
    path = os.path.join(cleaned_folder, fname)
    if not os.path.exists(path):
        print(f"Missing file: {fname}")
        return

    y, _ = librosa.load(path, sr=sampling_rate)

    n_fft = int(frame_length * sampling_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    spec_frames = log_mel.shape[1]

    # Prepare time axis
    time_axis = librosa.frames_to_time(np.arange(spec_frames), sr=sampling_rate, hop_length=hop_length)

    # Plot grid
    num_features = features.shape[2]
    n_rows = 11
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 25))
    axes = axes.flatten()

    for feat_idx in range(num_features):
        if feat_idx >= len(axes):
            break

        ax = axes[feat_idx]
        img = librosa.display.specshow(
            log_mel,
            sr=sampling_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )

        feat_vals = features[cycle_idx, :, feat_idx]
        x_feat = np.linspace(0, 1, len(feat_vals))
        x_spec = np.linspace(0, 1, spec_frames)
        aligned_feat = np.interp(x_spec, x_feat, feat_vals)

        ax_twin = ax.twinx()
        color = label_colors.get(label, "gray")
        ax_twin.plot(time_axis, aligned_feat, color=color, linewidth=1.5)
        ax.set_title(f"{feature_names[feat_idx]}")
        ax_twin.set_yticks([])

    # Remove empty subplots if any
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    fig.colorbar(img, ax=axes.tolist(), format="%+2.0f dB", orientation="vertical", fraction=0.02, pad=0.01)
    plt.suptitle(f"Cycle {cycle_idx} - {fname} - {label}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def perform_pca_and_tsne(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    pca_components: Union[int, float],
    tsne_components: int,
    perplexity: int,
    feature_names: list = None,
    plot_feature_contribution: bool = True
) -> None:
    """
    Apply PCA and t-SNE on (cycles, features) matrix and visualize results.

    Args:
        feature_matrix (np.ndarray): Feature matrix (cycles, features).
        labels (np.ndarray): Class labels for each cycle.
        pca_components (int or float): Number of components (int) or variance ratio (float) for PCA.
        tsne_components (int): Number of output dimensions for t-SNE (2 or 3).
        perplexity (int): Perplexity for t-SNE.
        feature_names (list, optional): Names of features (length = features).
        plot_feature_contribution (bool): Whether to plot PCA feature contributions.
    """
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # PCA
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(scaled)
    explained_var = np.sum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])
    pca_df["Label"] = labels

    # t-SNE
    tsne = TSNE(n_components=tsne_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=[f"t-SNE {i+1}" for i in range(tsne_components)])
    tsne_df["Label"] = labels

    # Color mapping
    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }
    palette = [class_colors.get(l, "gray") for l in labels]

    # PCA 2D plot
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], hue=labels, palette=class_colors, alpha=0.7)
        plt.title(f"PCA 2D Visualization (Explained variance: {explained_var:.2%})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # t-SNE 2D plot
    if tsne_components >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne_df["t-SNE 1"], y=tsne_df["t-SNE 2"], hue=labels, palette=class_colors, alpha=0.7)
        plt.title("t-SNE 2D Visualization")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # t-SNE 3D plot
    if tsne_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_df["t-SNE 1"], tsne_df["t-SNE 2"], tsne_df["t-SNE 3"], c=palette, alpha=0.7)
        ax.set_title("t-SNE 3D Visualization")
        plt.tight_layout()
        plt.show()

    # PCA contributions
    if plot_feature_contribution and feature_names is not None:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        contribution = np.sum(np.abs(loadings), axis=1)
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Total_Abs_Contribution": contribution
        }).sort_values("Total_Abs_Contribution", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=contrib_df, x="Feature", y="Total_Abs_Contribution", palette="viridis")
        plt.title("Total Contribution to PCA by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Total Abs Contribution")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.show()

def plot_feature_distributions_by_device(
    features: np.ndarray,
    devices: np.ndarray,
    feature_names: list
) -> None:
    """Plot KDE and boxplot grids of all features grouped by device.

    Args:
        features (np.ndarray): Array of shape (cycles, features).
        devices (np.ndarray): Array of shape (cycles,) with device names.
        feature_names (list): List of feature names (length = features.shape[1]).

    Returns:
        None: Displays the grid plots.
    """
    num_cycles, num_features = features.shape
    assert len(feature_names) == num_features, "Number of feature names must match features.shape[1]"

    device_colors = {
        "AKGC417L": "#1f77b4",
        "LittC2SE": "#ff7f0e",
        "Litt3200": "#2ca02c",
        "Meditron": "#d62728"
    }
    unique_devices = np.unique(devices)

    nrows, ncols = 11, 4

    # KDE Plots
    fig_kde, axes_kde = plt.subplots(nrows, ncols, figsize=(20, 30))
    axes_kde = axes_kde.flatten()

    for i in range(len(axes_kde)):
        ax = axes_kde[i]
        if i < num_features:
            for device in unique_devices:
                color = device_colors.get(device, "#333333")
                sns.kdeplot(
                    features[devices == device, i],
                    label=str(device) if i == 0 else None, 
                    fill=True,
                    color=color,
                    ax=ax
                )
            ax.set_title(feature_names[i])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.axis("off")

    handles, legend_labels = axes_kde[0].get_legend_handles_labels()
    fig_kde.legend(handles, legend_labels, title="Device", loc="upper center", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Boxplots
    fig_box, axes_box = plt.subplots(nrows, ncols, figsize=(20, 30))
    axes_box = axes_box.flatten()

    for i in range(len(axes_box)):
        ax = axes_box[i]
        if i < num_features:
            data = []
            group_labels = []
            for device in unique_devices:
                values = features[devices == device, i]
                data.append(values)
                group_labels.extend([str(device)] * len(values))
            data_flat = np.concatenate(data)
            palette = [device_colors.get(dev, "#333333") for dev in unique_devices]
            sns.boxplot(x=group_labels, y=data_flat, palette=palette, ax=ax)
            ax.set_title(feature_names[i])
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def perform_pca_and_tsne_by_device(
    feature_matrix: np.ndarray,
    devices: np.ndarray,
    pca_components: float,
    tsne_components: int,
    perplexity: int,
    feature_names: list = None,
    plot_feature_contribution: bool = True
) -> None:
    """
    Apply PCA and t-SNE on (cycles, features) matrix and visualize results, grouped by device.

    Args:
        feature_matrix (np.ndarray): Feature matrix (cycles, features).
        devices (np.ndarray): Device names per cycle.
        pca_components (float): Fraction of variance to retain in PCA (e.g., 0.95).
        tsne_components (int): Number of output dimensions for t-SNE (2 or 3).
        perplexity (int): Perplexity for t-SNE.
        feature_names (list, optional): Names of features (length = features.shape[1]).
        plot_feature_contribution (bool): Whether to plot PCA feature contributions.
    """
    X = feature_matrix

    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])
    pca_df["Device"] = devices

    # t-SNE
    tsne = TSNE(n_components=tsne_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=[f"t-SNE {i+1}" for i in range(tsne_components)])
    tsne_df["Device"] = devices

    # Color mapping
    device_colors = {
        "AKGC417L": "#1f77b4",
        "LittC2SE": "#ff7f0e",
        "Litt3200": "#2ca02c",
        "Meditron": "#d62728"
    }

    # PCA 2D
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=pca_df["PC1"], y=pca_df["PC2"],
            hue=pca_df["Device"], palette=device_colors, alpha=0.7
        )
        plt.title(f"PCA 2D Visualization (Variance: {pca_components})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # t-SNE 2D
    if tsne_components >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=tsne_df["t-SNE 1"], y=tsne_df["t-SNE 2"],
            hue=tsne_df["Device"], palette=device_colors, alpha=0.7
        )
        plt.title("t-SNE 2D Visualization by Device")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # t-SNE 3D
    if tsne_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = [device_colors.get(dev, "gray") for dev in tsne_df["Device"]]
        ax.scatter(
            tsne_df["t-SNE 1"], tsne_df["t-SNE 2"], tsne_df["t-SNE 3"],
            c=colors, alpha=0.7
        )
        ax.set_title("t-SNE 3D Visualization by Device")
        plt.tight_layout()
        plt.show()

    # PCA feature contribution
    if plot_feature_contribution and feature_names is not None:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        contribution = np.sum(np.abs(loadings), axis=1)
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Total_Abs_Contribution": contribution
        }).sort_values("Total_Abs_Contribution", ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=contrib_df, x="Feature", y="Total_Abs_Contribution", palette="viridis")
        plt.title("Total Contribution to PCA by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Total Abs Contribution")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.show()

def plot_signals_after_preprocessing(train_df, corrected_folder, raw_folder, sampling_rate, hop_size, k):
    """
    Plots raw and spectrum-corrected waveforms and spectrograms for the first k training files.

    Args:
        train_df (pd.DataFrame): DataFrame with at least a "file" column containing filenames.
        corrected_folder (str): Path to folder with corrected .wav files.
        raw_folder (str): Path to folder with raw/cleaned .wav files.
        sampling_rate (int): Sampling rate of audio.
        hop_size (int): Hop size for STFT.
        k (int): Number of files to plot.

    Returns:
        None
    """
    # Επιλογή των πρώτων k αρχείων ανεξαρτήτως συσκευής
    selected_files = train_df["file"].tolist()[:k]

    for fname in selected_files:
        raw_path = os.path.join(raw_folder, fname)
        corrected_path = os.path.join(corrected_folder, fname)

        raw_signal, _ = librosa.load(raw_path, sr=sampling_rate)
        corrected_signal, _ = librosa.load(corrected_path, sr=sampling_rate)

        # Spectrograms
        D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(raw_signal, hop_length=hop_size)), ref=np.max)
        D_corrected = librosa.amplitude_to_db(np.abs(librosa.stft(corrected_signal, hop_length=hop_size)), ref=np.max)

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(14, 6))

        # Waveforms
        axs[0, 0].plot(np.arange(len(raw_signal)) / sampling_rate, raw_signal, color="steelblue")
        axs[0, 0].set_title(f"Raw Signal - {fname}")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Amplitude")

        axs[0, 1].plot(np.arange(len(corrected_signal)) / sampling_rate, corrected_signal, color="orange")
        axs[0, 1].set_title(f"Preprocessed Signal - {fname}")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Amplitude")

        # Spectrograms
        img1 = librosa.display.specshow(D_raw, sr=sampling_rate, hop_length=hop_size, x_axis="time", y_axis="linear", ax=axs[1, 0])
        axs[1, 0].set_title("Raw Spectrogram")
        fig.colorbar(img1, ax=axs[1, 0], format="%+2.0f dB")

        img2 = librosa.display.specshow(D_corrected, sr=sampling_rate, hop_length=hop_size, x_axis="time", y_axis="linear", ax=axs[1, 1])
        axs[1, 1].set_title("Preprocessed Spectrogram")
        fig.colorbar(img2, ax=axs[1, 1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()

def plot_signals_after_domain_adaptation(file_list, cleaned_folder, corrected_folder, sampling_rate, hop_size):
    """
    Plots the waveform and spectrogram of raw and spectrum-corrected audio files.

    Args:
        file_list (list of str): List of filenames to visualize.
        cleaned_folder (str): Folder containing the initial preprocessed (uncorrected) .wav files.
        corrected_folder (str): Folder with spectrum-corrected .wav files.
        sampling_rate (int): Sampling rate of audio files.
        hop_size (int): Hop size used for spectrogram computation.

    Returns:
        None
    """
    for fname in file_list:
        preprocessed_path = os.path.join(cleaned_folder, fname)
        corrected_path = os.path.join(corrected_folder, fname)

        # Load both initial preprocessed and corrected files
        preprocessed_signal, _ = librosa.load(preprocessed_path, sr=sampling_rate)
        corrected_signal, _ = librosa.load(corrected_path, sr=sampling_rate)

        # Compute spectrograms
        D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(preprocessed_signal, hop_length=hop_size)), ref=np.max)
        D_corrected = librosa.amplitude_to_db(np.abs(librosa.stft(corrected_signal, hop_length=hop_size)), ref=np.max)

        fig, axs = plt.subplots(2, 2, figsize=(14, 6))

        # Waveforms
        axs[0, 0].plot(np.arange(len(preprocessed_signal)) / sampling_rate, preprocessed_signal, color="steelblue")
        axs[0, 0].set_title(f"Preprocessed Signal - {fname}")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Amplitude")

        axs[0, 1].plot(np.arange(len(corrected_signal)) / sampling_rate, corrected_signal, color="orange")
        axs[0, 1].set_title(f"Corrected Signal - {fname}")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Amplitude")

        # Spectrograms
        img1 = librosa.display.specshow(
            D_raw,
            sr=sampling_rate,
            hop_length=hop_size,
            x_axis="time",
            y_axis="linear",
            ax=axs[1, 0]
        )
        axs[1, 0].set_title("Preprocessed Spectrogram")
        fig.colorbar(img1, ax=axs[1, 0], format="%+2.0f dB")

        img2 = librosa.display.specshow(
            D_corrected,
            sr=sampling_rate,
            hop_length=hop_size,
            x_axis="time",
            y_axis="linear",
            ax=axs[1, 1]
        )
        axs[1, 1].set_title("Corrected Spectrogram")
        fig.colorbar(img2, ax=axs[1, 1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()

def plot_mean_spectra_per_device(
    folder: str,
    df,
    devices: list,
    sampling_rate: int,
    frame_size: int,
    hop_size: int
) -> None:
    """
    Plot mean spectrum per class for each device in a 2x2 grid using predefined class colors.

    Args:
        folder (str): Path to folder containing raw .wav cycle files.
        df (pd.DataFrame): DataFrame containing at least 'file' and 'class' columns.
        devices (list): List of device names (str) to include.
        sampling_rate (int): Sampling rate to load audio files.
        frame_size (int): Frame size (n_fft) for STFT.
        hop_size (int): Hop length for STFT.

    Returns:
        None: Displays a 2x2 grid of plots showing the mean power spectrum per class for each device.
    """
    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }

    device_class_signals = {device: defaultdict(list) for device in devices}
    file_to_label = dict(zip(df["file"], df["class"]))

    # Group signals by device and class
    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):
            continue
        device = next((d for d in devices if d in fname), None)
        label = file_to_label.get(fname)
        if device and label:
            path = os.path.join(folder, fname)
            y, _ = librosa.load(path, sr=sampling_rate)
            if not np.isnan(y).any():
                device_class_signals[device][label].append(y)

    # Plot in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, device in enumerate(devices):
        ax = axes[i]
        for cls, signals in device_class_signals[device].items():
            all_psd = []
            for y in signals:
                S = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_size))**2
                S_db = librosa.power_to_db(S, ref=np.max)
                mean_spectrum = np.mean(S_db, axis=1)
                all_psd.append(mean_spectrum)
            if all_psd:
                avg_psd = np.mean(np.stack(all_psd), axis=0)
                freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=frame_size)
                ax.plot(freqs, avg_psd, label=cls, color=class_colors.get(cls, "#333333"))

        ax.set_title(f"Mean Spectrum – {device}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_feature_selection_effect(X, y, feature_names, n_neighbors=3, n_splits=5):
    """
    Performs iterative feature selection with kNN and cross-validation using NumPy arrays.

    Args:
        X (np.ndarray): Input features of shape (samples, features).
        y (np.ndarray): Labels.
        feature_names_sorted (list): Feature names ordered most to least important.
        n_neighbors (int): Number of neighbors for kNN.
        n_splits (int): Number of CV folds.

    Returns:
        list: Number of features at each step.
        list: Mean F1 macro scores.
    """
    # Map feature names to indices
    feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}

    features_current = feature_names.copy()
    n_features_list = []
    f1_scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')

    while len(features_current) >= 1:
        # Get indices of current features
        indices = [feature_name_to_idx[name] for name in features_current]
        X_sel = X[:, indices]

        # kNN model + cross-validation
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(knn, X_sel, y, cv=skf, scoring=scorer, n_jobs=-1)

        mean_f1 = np.mean(scores)
        f1_scores.append(mean_f1)
        n_features_list.append(len(features_current))

        # Remove last 2 features
        if len(features_current) > 1:
            features_current = features_current[:-2]
        else:
            break

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(n_features_list, f1_scores, marker='o')
    plt.xlabel("Feature Vector Length")
    plt.ylabel("F1 Macro Avg Score")
    plt.title("F1 Macro Avg Score vs Number of Features")
    plt.show()

def plot_dataset_spectrograms(audio_folder, k, sr, hop_length):
    """
    Plots the first k spectrograms with vertical lines showing respiratory cycle boundaries.

    Args:
        audio_folder (str): Folder with .wav and .txt files.
        k (int): Number of files to plot.
        sr (int): Sampling rate.
        hop_length (int): Hop size for STFT.
    """
    wav_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])[:k]

    for fname in wav_files:
        wav_path = os.path.join(audio_folder, fname)
        txt_path = wav_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            continue

        # Load audio
        y, _ = librosa.load(wav_path, sr=sr)

        # Compute spectrogram with user-defined hop_length
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)

        # Plot
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            S, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='linear', cmap='magma'
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(fname)

        # Load annotations (start, end, crackle, wheeze)
        annotations = np.loadtxt(txt_path)
        if annotations.ndim == 1:
            annotations = annotations[None, :]

        for row in annotations:
            start, end, *_ = row
            plt.axvline(start, color='white', linestyle='--', linewidth=1)
            plt.axvline(end, color='white', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.show()

def plot_average_spectrum_per_class(
    train_df: pd.DataFrame,
    raw_cycle_folder: str,
    sampling_rate: int,
    frame_size: int,
    hop_size: int
) -> None:
    """Plots the average power spectrum per class and highlights the region
    with the highest spectral energy.

    Args:
        train_df (pd.DataFrame): DataFrame containing file names and class labels.
        raw_cycle_folder (str): Path to folder containing the cycle .wav files.
        sampling_rate (int): Sampling rate of the audio signals.
        frame_size (int): STFT window size (n_fft).
        hop_size (int): Hop length for STFT.

    Returns:
        None
    """
    spectra_per_class: Dict[str, list] = defaultdict(list)

    # Define fixed colors per class
    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e",
        "Both": "#219ebc"
    }

    # Compute power spectrum for each file and group by class
    for _, row in train_df.iterrows():
        file_path = os.path.join(raw_cycle_folder, row["file"])
        y, sr = librosa.load(file_path, sr=sampling_rate)

        S = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_size)) ** 2
        psd = np.mean(S, axis=1)  # average power spectrum
        spectra_per_class[row["class"]].append(psd)

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=frame_size)

    # Compute mean PSD across all classes to find peak region
    all_psd = np.concatenate(list(spectra_per_class.values()), axis=0)
    mean_total_psd = np.mean(all_psd, axis=0)

    # Identify region with top 5% power values
    threshold = np.percentile(mean_total_psd, 95)
    mask = mean_total_psd >= threshold
    highlight_start = freqs[np.where(mask)[0][0]]
    highlight_end = freqs[np.where(mask)[0][-1]]

    # Plot spectra
    plt.figure(figsize=(10, 6))
    for cls, psd_list in spectra_per_class.items():
        avg_psd = np.mean(psd_list, axis=0)
        plt.plot(freqs, avg_psd, label=cls, color=label_colors.get(cls, None))

    # Highlight high-power region
    plt.axvspan(highlight_start, highlight_end, color='orange', alpha=0.3,
                label=f"Top 5% Power Region ({int(highlight_start)}–{int(highlight_end)} Hz)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean Power Spectrum")
    plt.title("Average Frequency Spectrum per Class")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

def plot_vtlp_effects(augmented_df, raw_folder, aug_folder, k=3, sr=4000, n_fft=256, hop_length=64, n_mels=64):
    """
    For each of the first k augmented samples:
    - Shows original & VTLP log-mel spectrograms side-by-side
    - Plots PSD curves underneath

    Args:
        augmented_df (pd.DataFrame): DataFrame with VTLP-augmented file info.
        raw_folder (str): Path to original wav files.
        aug_folder (str): Path to VTLP-augmented wav files.
        k (int): Number of samples to visualize.
        sr (int): Sampling rate.
        n_fft (int): FFT size for STFT and PSD.
        hop_length (int): Hop size for spectrogram.
        n_mels (int): Number of mel bands.
    """

    for i in range(min(k, len(augmented_df))):
        aug_row = augmented_df.iloc[i]
        aug_fname = aug_row["file"]
        orig_fname = aug_fname.replace("AUG_", "").split("_vtlp")[0] + ".wav"

        path_orig = os.path.join(raw_folder, orig_fname)
        path_aug = os.path.join(aug_folder, aug_fname)

        y_orig, _ = librosa.load(path_orig, sr=sr)
        y_aug, _ = librosa.load(path_aug, sr=sr)

        # Compute Log-Mel Spectrograms 
        S_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_aug = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        log_S_orig = librosa.power_to_db(S_orig, ref=np.max)
        log_S_aug = librosa.power_to_db(S_aug, ref=np.max)

        # Compute PSD 
        f1, Pxx_orig = scipy.signal.welch(y_orig, fs=sr, nperseg=n_fft)
        f2, Pxx_aug = scipy.signal.welch(y_aug, fs=sr, nperseg=n_fft)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

        # Spectrograms 
        librosa.display.specshow(log_S_orig, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[0][0])
        axes[0][0].set_title(f"Original: {orig_fname}")

        librosa.display.specshow(log_S_aug, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[0][1])
        axes[0][1].set_title(f"Augmented: {aug_fname}")

        # PSD comparison 
        axes[1][0].semilogy(f1, Pxx_orig, label="Original", linewidth=2)
        axes[1][0].semilogy(f2, Pxx_aug, label="VTLP", linewidth=2)
        axes[1][0].set_title("Power Spectrum")
        axes[1][0].set_xlabel("Frequency (Hz)")
        axes[1][0].set_ylabel("Power")
        axes[1][0].legend()
        axes[1][1].axis('off')  

        plt.suptitle(f"VTLP Effect Visualization - Sample {i+1} of Training set", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_respiratory_cycles_after_length_fix(
    original_folder: str,
    fixed_folder: str,
    k: int,
    sr: int
):
    """
    Plot original vs fixed-length respiratory cycles.
    """
    plotted = 0

    for fixed_fname in sorted(os.listdir(fixed_folder)):
        if not fixed_fname.endswith(".wav"):
            continue

        orig_fname = fixed_fname.replace(".wav", "_original.wav")
        fixed_path = os.path.join(fixed_folder, fixed_fname)
        orig_path  = os.path.join(original_folder, orig_fname)

        if not os.path.exists(orig_path):
            continue

        # Load cycles
        y_orig, _  = librosa.load(orig_path,  sr=sr, mono=True)
        y_fixed, _ = librosa.load(fixed_path, sr=sr, mono=True)

        # Plot
        t_orig  = np.linspace(0, len(y_orig)  / sr, len(y_orig),  endpoint=False)
        t_fixed = np.linspace(0, len(y_fixed) / sr, len(y_fixed), endpoint=False)

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(t_orig, y_orig, color='steelblue')
        plt.title(f"{os.path.splitext(fixed_fname)[0]} (original)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        plt.plot(t_fixed, y_fixed, color='darkorange')
        plt.title(f"{os.path.splitext(fixed_fname)[0]} (fixed)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        plotted += 1
        if plotted >= k:
            return

def plot_spectrograms_after_highpass_filter(
    raw_folder: str,
    filtered_folder: str,
    k: int,
    sr: int = 4000,
    n_mels: int = 64,
    hop_length: int = 256,
):
    """
    Plot log-Mel spectrograms before vs after the high-pass filter.

    Args:
        raw_folder: Path to files pre-filter.
        filtered_folder: Path to files post-filter.
        k: Number of pairs to display.
        sr: Sampling rate to load/resample.
        n_mels: Number of Mel bands.
        hop_length: Hop length for STFT.
    """
    def logmel_db(y: np.ndarray) -> np.ndarray:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, power=2.0)
        return librosa.power_to_db(S, ref=1.0)

    plotted = 0
    for fname in sorted(os.listdir(raw_folder)):
        if not fname.endswith(".wav"):
            continue

        raw_path = os.path.join(raw_folder, fname)
        filt_path = os.path.join(filtered_folder, fname)
        if not os.path.exists(filt_path):
            continue

        # Load 
        y_raw, _  = librosa.load(raw_path, sr=sr, mono=True)
        y_filt, _ = librosa.load(filt_path, sr=sr, mono=True)

        # Compute log-Mel in dB
        D_raw  = logmel_db(y_raw)
        D_filt = logmel_db(y_filt)

        # Common color scale for fair comparison
        vmin = min(D_raw.min(), D_filt.min())
        vmax = max(D_raw.max(), D_filt.max())

        # Plot
        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 1)
        librosa.display.specshow(D_raw, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma', vmin=vmin, vmax=vmax)
        plt.title(f"{fname} (original)")
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(1, 2, 2)
        librosa.display.specshow(D_filt, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='magma', vmin=vmin, vmax=vmax)
        plt.title(f"{fname} (after high-pass filter)")
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.show()

        plotted += 1
        if plotted >= k:
            return

def plot_respiratory_cycles_after_amplitude_normalization(
    raw_folder: str,
    cleaned_folder: str,
    k: int,
    sr: int):
    """
    Plot each respiratory cycle before and after amplitude normalization 

    Args:
        raw_folder (str): Path to files before normalization.
        cleaned_folder (str): Path to files after normalization.
        k (int): Number of total respiratory cycles to visualize.
        sr (int): Sampling rate.
    """
    plotted = 0

    for fname in sorted(os.listdir(raw_folder)):
        if not fname.endswith(".wav"):
            continue

        raw_path = os.path.join(raw_folder, fname)
        clean_path = os.path.join(cleaned_folder, fname)
        if not os.path.exists(clean_path):
            continue

        # Load both versions
        y_fixed_raw, _ = librosa.load(raw_path, sr=sr, mono=True)
        y_fixed_norm, _ = librosa.load(clean_path, sr=sr, mono=True)

        # Create time axis
        t = np.linspace(0, 3, len(y_fixed_raw), endpoint=False)

        # Plot
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(t, y_fixed_raw, color='steelblue')
        plt.title(f"{fname} (fixed-length)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        plt.plot(t, y_fixed_norm, color='darkorange')
        plt.title(f"{fname} (normalized)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        plotted += 1
        if plotted >= k:
            return