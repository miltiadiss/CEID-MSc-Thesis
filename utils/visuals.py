from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
import librosa
import librosa.display
import scipy.signal
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import os
from utils.audio_preprocessing_utils import classify
from collections import defaultdict
from typing import Dict

# === Functions ===
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

def plot_cycle_duration_distribution(annotation_folder: str) -> None:
    """
    Plots the histogram of respiratory cycle durations using annotation files.

    Args:
        annotation_folder (str): Path to the folder containing .txt annotation files.

    Returns:
        None
    """
    # Collect durations from annotation files
    cycle_durations = []

    for file in os.listdir(annotation_folder):
        if file.endswith(".txt"):
            txt_path = os.path.join(annotation_folder, file)
            annotations = np.loadtxt(txt_path)

            for (start, end, crackle, wheeze) in annotations:
                duration = end - start
                if duration > 0:
                    cycle_durations.append(duration)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(cycle_durations, bins=50, color='steelblue', edgecolor='black')
    plt.title("Distribution of Respiratory Cycle Lengths")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
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

def plot_feature_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    n_frames: int,
    n_features: int,
    n_cycles_to_plot: int = 5
) -> None:
    """
    For each feature, plots:
        1. Frame-level values for selected cycles.
        2. Average per-frame evolution grouped by class.
        3. F-value and -log10(p-value) across frames.

    Args:
        features (np.ndarray): Feature matrix of shape (num_cycles, n_frames * n_features).
        labels (np.ndarray): Array of string labels per cycle (e.g., "Crackle", "Wheeze").
        feature_names (list): List of feature names of length `n_features`.
        n_frames (int): Number of frames per cycle.
        n_features (int): Number of features per frame.
        n_cycles_to_plot (int): How many individual cycles to show in (1).

    Returns:
        None
    """
    assert features.shape[1] == n_frames * n_features, "Feature dimensions do not match n_frames × n_features"

    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e",
        "Both": "#219ebc"
    }

    # Reshape (cycles, frames x features) to (cycles, frames, features)
    reshaped = features.reshape(-1, n_frames, n_features)
    n_samples = reshaped.shape[0]

    # Compute F-values and p-values on reshaped data
    f_values = np.zeros((n_frames, n_features))
    p_values = np.zeros_like(f_values)

    for f in range(n_features):
        X_feat = reshaped[:, :, f]  # (samples, frames)
        for t in range(n_frames):
            f_val, p_val = f_classif(X_feat[:, [t]], labels)
            f_values[t, f] = f_val
            p_values[t, f] = p_val

    # Get selected cycles to plot
    selected_indices = np.arange(min(n_cycles_to_plot, n_samples))

    for i, name in enumerate(feature_names):
        # Plot 1: Evolution of features over frmaes for selected cycles 
        fig, ax = plt.subplots(figsize=(10, 4))
        used_labels = set()

        for c in selected_indices:
            label_c = labels[c]
            color = label_colors.get(label_c, "gray")
            show_label = label_c if label_c not in used_labels else None
            ax.plot(range(n_frames), reshaped[c, :, i], label=show_label, color=color)
            used_labels.add(label_c)

        ax.set_title(f"[{name}] Evolution over frames (first {n_cycles_to_plot} cycles)")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Feature Value")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: Mean evolution of features per label 
        plt.figure(figsize=(10, 4))
        for label in np.unique(labels):
            mask = labels == label
            mean_curve = reshaped[mask, :, i].mean(axis=0)
            color = label_colors.get(label, "gray")
            plt.plot(mean_curve, label=label, color=color)

        plt.title(f"[{name}] Mean Evolution per Class")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean Feature Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 3: F - values and p - values 
        log_p_values = -np.log10(p_values + 1e-8) # Log over p - values to handle the difference in scaling

        plt.figure(figsize=(10, 4))
        plt.plot(f_values[:, i], label="F-value", color="blue")
        plt.plot(log_p_values[:, i], label="-log10(p-value)", color="red")
        plt.title(f"[{name}] F-score and -log10(p-value) across Frames")
        plt.xlabel("Frame Index")
        plt.ylabel("Statistical Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_feature_on_spectrogram(
    fname,
    feature_vector,
    label,
    feature_index,
    num_features,
    sampling_rate,
    frame_length,
    hop_length,
    n_mels
):
    """
    Plot aligned spectrogram and feature vector for a single respiratory cycle.
    """
    label_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e",
        "Both": "#219ebc"
    }

    cleaned_folder = "out/cleaned_audio_cycles"
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

    feat_vals = feature_vector[feature_index::num_features]
    feat_frames = len(feat_vals)

    # Resample feature vector to match number of spectrogram frames
    x_feat = np.linspace(0, 1, feat_frames)
    x_spec = np.linspace(0, 1, spec_frames)
    aligned_feat_vals = np.interp(x_spec, x_feat, feat_vals)

    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.set_facecolor('white')

    img = librosa.display.specshow(
        log_mel,
        sr=sampling_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_ylabel("Mel Frequency (Hz)")

    # Overlay feature on spectrogram
    ax_twin = ax.twinx()
    color = label_colors.get(label, "gray")
    time_axis = librosa.frames_to_time(np.arange(spec_frames), sr=sampling_rate, hop_length=hop_length)
    ax_twin.plot(time_axis, aligned_feat_vals, color=color, linewidth=2, label=label)
    ax_twin.set_ylabel("Feature Value", color=color)
    ax_twin.tick_params(axis='y', labelcolor=color)
    ax.set_title(f"{fname} - Feature {feature_index}")
    ax_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    num_frames: int,
    label_palette: str = "Set2"
) -> None:
    """
    Plot KDE distributions and boxplots of each feature (frame-wise) grouped by label.

    Args:
        features (np.ndarray): Feature matrix (samples x frames × features).
        labels (np.ndarray): Labels array (samples,).
        feature_names (list): Names of the features.
        num_frames (int): Number of frames per sample (cycle).
        label_palette (str): Color palette for seaborn plots.

    Returns:
        None
    """
    num_samples = features.shape[0]
    num_features = features.shape[1] // num_frames

    assert num_features == len(feature_names), "Number of feature names doesn't match features per frame"

    # Reshape to (samples, frames, features)
    features_reshaped = features.reshape(num_samples, num_frames, num_features)

    # Flatten to (samples × frames, features)
    features_flat = features_reshaped.reshape(-1, num_features)

    # Expand labels: repeat each label num_frames times
    labels_flat = np.repeat(labels, num_frames)

    unique_labels = np.unique(labels)
    palette = sns.color_palette(label_palette, len(unique_labels))

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(16, 6))

        # KDE Plot
        plt.subplot(1, 2, 1)
        for label, color in zip(unique_labels, palette):
            mask = labels_flat == label
            sns.kdeplot(features_flat[mask, i], label=str(label), fill=True, color=color)
        plt.title(f"[{feature_name}] KDE Plot")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend(title="Class")

        # Boxplot
        plt.subplot(1, 2, 2)
        data = []
        group_labels = []
        for label in unique_labels:
            mask = labels_flat == label
            data.append(features_flat[mask, i])
            group_labels.extend([str(label)] * np.sum(mask))

        data_flat = np.concatenate(data)
        sns.boxplot(x=group_labels, y=data_flat, palette=palette)
        plt.title(f"[{feature_name}] Boxplot")
        plt.xlabel("Class")
        plt.ylabel(feature_name)

        plt.tight_layout()
        plt.show()

def perform_pca_and_tsne(
    feature_matrix: Union[pd.DataFrame, np.ndarray],
    labels: Union[pd.Series, list, np.ndarray],
    pca_components: Union[int, float],
    tsne_components: int,
    perplexity: int,
    n_features: int,
    feature_names: list = None,
    plot_feature_contribution = True
) -> None:
    """
    Apply PCA and t-SNE on the given feature matrix and visualize the results.
    Also plots the total contribution of each original feature to the PCA axes.

    Args:
        feature_matrix (pd.DataFrame or np.ndarray): Feature values (rows = samples).
        labels (Union[pd.Series, list, np.ndarray]): Class labels for each sample.
        pca_components (int or float): Number of components (int) or variance ratio (float) to retain in PCA.
        tsne_components (int): Number of output dimensions for t-SNE (2 or 3).
        perplexity (int): Perplexity parameter for t-SNE.
        n_features (int): Number of features per frame.
        feature_names (list, optional): Names of original features (length n_features).
    """
    # If input is DataFrame
    if isinstance(feature_matrix, np.ndarray):
        X = feature_matrix
    else:
        X = feature_matrix.values

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])
    pca_df["Label"] = labels

    # t-SNE
    tsne = TSNE(n_components=tsne_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=[f"t-SNE {i+1}" for i in range(tsne_components)])
    tsne_df["Label"] = labels

    # Visualization
    fig = plt.figure(figsize=(18, 10))
    palette = sns.color_palette("Set2", len(set(labels)))
    label_mapping = {label: palette[i] for i, label in enumerate(set(labels))}
    colors_pca = [label_mapping[label] for label in labels]
    colors_tsne = [label_mapping[label] for label in labels]

    # PCA 2D
    ax1 = fig.add_subplot(2, 2, 1)
    sns.scatterplot(ax=ax1, x=pca_df["PC1"], y=pca_df["PC2"], hue=labels, palette=palette, alpha=0.7)
    ax1.set_title("PCA 2D Visualization")

    # t-SNE 2D
    ax2 = fig.add_subplot(2, 2, 2)
    sns.scatterplot(ax=ax2, x=tsne_df["t-SNE 1"], y=tsne_df["t-SNE 2"], hue=labels, palette=palette, alpha=0.7)
    ax2.set_title("t-SNE 2D Visualization")

    # PCA 3D
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax3d.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], c=colors_pca, alpha=0.7)
    ax3d.set_title("PCA 3D Visualization")

    # t-SNE 3D
    ax4d = fig.add_subplot(2, 2, 4, projection='3d')
    ax4d.scatter(tsne_df["t-SNE 1"], tsne_df["t-SNE 2"], tsne_df["t-SNE 3"], c=colors_tsne, alpha=0.7)
    ax4d.set_title("t-SNE 3D Visualization")

    plt.tight_layout()
    plt.show()

    # PCA Feature Contribution Aggregation 
    if(plot_feature_contribution==True):
        total_dims = X.shape[1]
        assert total_dims % n_features == 0, "Feature dimensions not divisible by n_features"
        n_frames = total_dims // n_features

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # shape: (frames × features, components)
        total_contribution = np.zeros(n_features)
        for feat_idx in range(n_features):
            indices = [f * n_features + feat_idx for f in range(n_frames)]
            total_contribution[feat_idx] = np.sum(np.abs(loadings[indices, :]))

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Total_Abs_Contribution": total_contribution
        }).sort_values("Total_Abs_Contribution", ascending=False)

        # Plot contribution
        plt.figure(figsize=(12, 6))
        sns.barplot(data=contrib_df, x="Feature", y="Total_Abs_Contribution", palette="viridis")
        plt.title("Total Contribution to PCA by Original Feature")
        plt.xlabel("Feature")
        plt.ylabel("Total Abs Contribution")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.show()

def evaluate_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    positive_label: str = "Wheeze",
    negative_label: str = "Crackle",
    n_splits: int = 5,
    random_state: int = 42
) -> None:
    """
    Perform stratified 5-fold cross-validation by training on X_train and testing on X_test.

    Args:
        X_train (np.ndarray): Feature matrix of training samples.
        y_train (np.ndarray): Labels for training samples.
        X_test (np.ndarray): Feature matrix of test samples.
        y_test (np.ndarray): Labels for test samples.
        positive_label (str): Label to treat as positive class.
        negative_label (str): Label to treat as negative class.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed.

    Returns:
        None. Prints evaluation metrics and plots ROC curves.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    # Filter only Crackle and Wheeze
    target_classes = [negative_label, positive_label]
    mask_train = np.isin(y_train, target_classes)
    mask_test = np.isin(y_test, target_classes)

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    label_map = {negative_label: 0, positive_label: 1}
    y_train_bin = np.array([label_map[l] for l in y_train])
    y_test_bin = np.array([label_map[l] for l in y_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    tprs, fprs, aucs = [], [], []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_bin), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train_bin[train_idx]
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X_tr, y_tr)

        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = clf.predict(X_test_scaled)

        all_y_true.extend(y_test_bin)
        all_y_pred.extend(y_pred)

        fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
        roc_auc = auc(fpr, tpr)
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(roc_auc)

        print(f"\nFold {fold} Report:")
        print(classification_report(y_test_bin, y_pred, target_names=target_classes))
        print(confusion_matrix(y_test_bin, y_pred))

    # Plot ROC
    plt.figure(figsize=(8, 5))
    for i in range(n_splits):
        plt.plot(fprs[i], tprs[i], alpha=0.3, label=f"Fold {i+1} AUC={aucs[i]:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set (Model trained on CV folds)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    print("\n=== Overall Performance on Test Set ===")
    print(classification_report(all_y_true, all_y_pred, target_names=target_classes))
    print("Average AUC:", np.mean(aucs))

def plot_signals_after_spectrum_correction(file_list, cleaned_folder, corrected_folder, sampling_rate, hop_size):
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