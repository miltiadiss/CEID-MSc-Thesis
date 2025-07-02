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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import os

# === Functions ===
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
    feature_names: list = None
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