from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif

# === Functions ===
def perform_pca_and_tsne(
    feature_matrix: pd.DataFrame,
    labels: Union[pd.Series, list],
    pca_components: int,
    tsne_components: int,
    perplexity: int
) -> None:
    """Apply PCA and t-SNE on the given feature matrix and visualize the results.

    Args:
        feature_matrix (pd.DataFrame): DataFrame containing feature values (rows = samples).
        labels (Union[pd.Series, list]): Class labels for each sample.
        pca_components (int): Number of components or variance ratio to retain in PCA.
        tsne_components (int): Number of output dimensions for t-SNE (2 or 3).
        perplexity (int): Perplexity parameter for t-SNE.

    Returns:
        None
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # PCA
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])])
    pca_df["Label"] = labels

    # t-SNE
    tsne = TSNE(n_components=tsne_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)
    tsne_df = pd.DataFrame(tsne_result, columns=[f"t-SNE {i+1}" for i in range(tsne_components)])
    tsne_df["Label"] = labels

    # PCA feature importance
    pca_feature_importance = pd.DataFrame(
        np.abs(pca.components_[:3]), columns=feature_matrix.columns
    ).mean(axis=0).sort_values(ascending=False)

    # t-SNE feature correlation
    correlation_matrix = np.corrcoef(scaled_features.T, tsne_result.T)
    tsne_feature_importance = pd.DataFrame(
        np.abs(correlation_matrix[:len(feature_matrix.columns), len(feature_matrix.columns):]),
        index=feature_matrix.columns,
        columns=[f"t-SNE {i+1}" for i in range(tsne_components)]
    )
    tsne_feature_importance["Mean Influence"] = tsne_feature_importance.mean(axis=1)
    tsne_feature_importance.sort_values(by="Mean Influence", ascending=False, inplace=True)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    palette = sns.color_palette("Set2", len(set(labels)))
    label_mapping = {label: palette[i] for i, label in enumerate(set(labels))}
    colors_pca = [label_mapping[label] for label in labels]
    colors_tsne = [label_mapping[label] for label in labels]

    sns.scatterplot(ax=axes[0, 0], x=pca_df["PC1"], y=pca_df["PC2"], hue=labels, palette=palette, alpha=0.7)
    axes[0, 0].set_title("PCA 2D Visualization")

    sns.scatterplot(ax=axes[0, 1], x=tsne_df["t-SNE 1"], y=tsne_df["t-SNE 2"], hue=labels, palette=palette, alpha=0.7)
    axes[0, 1].set_title("t-SNE 2D Visualization")

    sns.barplot(ax=axes[0, 2], x=pca_feature_importance.values, y=pca_feature_importance.index, palette="coolwarm")
    axes[0, 2].set_title("Feature Contributions to PCA")

    ax_pca_3d = fig.add_subplot(2, 3, 4, projection='3d')
    ax_pca_3d.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], c=colors_pca, alpha=0.7)
    ax_pca_3d.set_title("PCA 3D Visualization")

    ax_tsne_3d = fig.add_subplot(2, 3, 5, projection='3d')
    ax_tsne_3d.scatter(tsne_df["t-SNE 1"], tsne_df["t-SNE 2"], tsne_df["t-SNE 3"], c=colors_tsne, alpha=0.7)
    ax_tsne_3d.set_title("t-SNE 3D Visualization")

    sns.barplot(ax=axes[1, 2], x=tsne_feature_importance["Mean Influence"], y=tsne_feature_importance.index, palette="coolwarm")
    axes[1, 2].set_title("Feature Contributions to t-SNE")

    plt.tight_layout()
    plt.show()

def plot_feature_distributions(
    df: pd.DataFrame,
    label_column: str,
    feature_names: list,
    label_palette: str = "Set2"
) -> None:
    """Visualize feature distributions using KDE and boxplots for each class.

    Args:
        df (pd.DataFrame): DataFrame containing features and class labels.
        label_column (str): Column name in df containing class labels.
        feature_names (list): List of features to visualize.
        label_palette (str): Seaborn color palette to use for plots.

    Returns:
        None
    """
    for feat in feature_names:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.kdeplot(data=df, x=feat, hue=label_column, fill=True, common_norm=False, alpha=0.4, palette=label_palette)
        plt.title(f"Distribution of '{feat}'")
        plt.xlabel(feat)
        plt.ylabel("Density")

        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x=label_column, y=feat, palette=label_palette)
        plt.title(f"Boxplot of '{feat}'")
        plt.xlabel("Label")
        plt.ylabel(feat)

        plt.tight_layout()
        plt.show()
        plt.close()

def remove_outliers_iqr(
    df: pd.DataFrame,
    exclude_columns: list = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Remove outliers using IQR filtering for all numeric columns except those excluded.

    Args:
        df (pd.DataFrame): Input dataframe with numeric features.
        exclude_columns (List[str], optional): Columns to exclude from filtering.
        verbose (bool): Whether to print IQR stats per column.

    Returns:
        pd.DataFrame: Filtered dataframe with outliers removed.
    """
    if exclude_columns is None:
        exclude_columns = []

    numeric_columns = [col for col in df.columns if col not in exclude_columns and np.issubdtype(df[col].dtype, np.number)]
    df_filtered = df.copy()

    for column in numeric_columns:
        Q1 = np.percentile(df_filtered[column].dropna(), 25)
        Q3 = np.percentile(df_filtered[column].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if verbose:
            outlier_count = ((df_filtered[column] < lower_bound) | (df_filtered[column] > upper_bound)).sum()
            print(f"[{column}] IQR -> Q1={Q1:.2f}, Q3={Q3:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")
            print(f"Outliers removed: {outlier_count}")
            print("-" * 40)

        df_filtered = df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)]

    return df_filtered

def plot_feature_correlation_heatmap(df: pd.DataFrame, feature_names: list, method: str = "pearson") -> None:
    """Plot a correlation heatmap between all feature pairs.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        feature_names (List[str]): List of feature column names.
        method (str, optional): Correlation method ('pearson', 'spearman', or 'kendall'). Defaults to "pearson".

    Returns:
        None
    """
    corr = df[feature_names].corr(method=method)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
    plt.title(f"Feature-to-Feature Correlation Heatmap ({method.capitalize()})")
    plt.tight_layout()
    plt.show()

def plot_feature_class_relevance(df: pd.DataFrame, feature_names: list, label_column: str) -> None:
    """Plot feature relevance to class labels using ANOVA F-score and Mutual Information.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and labels.
        feature_names (List[str]): List of feature column names.
        label_column (str): Column name for the class labels.

    Returns:
        None
    """
    X = df[feature_names]
    y = df[label_column]

    # ANOVA F-score
    f_scores, _ = f_classif(X, y)
    f_df = pd.Series(f_scores, index=feature_names).sort_values(ascending=False)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, discrete_features=False)
    mi_df = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    f_df.plot(kind='bar', color='skyblue')
    plt.title("ANOVA F-score (Feature Relevance)")
    plt.ylabel("F-value")

    plt.subplot(1, 2, 2)
    mi_df.plot(kind='bar', color='salmon')
    plt.title("Mutual Information (Feature Relevance)")
    plt.ylabel("MI Score")

    plt.tight_layout()
    plt.show()