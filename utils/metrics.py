import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# === Functions ===
def coral_distance(Xs, Xt):
    """
    Computes the CORAL (CORrelation ALignment) distance between two feature domains.

    Args:
        Xs: np.ndarray, source domain features (n_samples x n_features)
        Xt: np.ndarray, target domain features (n_samples x n_features)

    Returns:
        float: CORAL distance value
    """
    # Center the features
    Xs = Xs - np.mean(Xs, axis=0)
    Xt = Xt - np.mean(Xt, axis=0)

    # Compute covariance matrices
    Cs = np.cov(Xs, rowvar=False)
    Ct = np.cov(Xt, rowvar=False)

    # Compute CORAL loss: Frobenius norm of the difference
    loss = np.sum((Cs - Ct) ** 2)
    return loss

import numpy as np

def mean_wasserstein(X, Y):
    """
    Compute the mean Wasserstein (Earth Mover's) distance across all feature dimensions.

    This function calculates the 1D Wasserstein distance (also known as Earth Mover's Distance)
    between corresponding features (columns) of two input datasets and returns the average distance.

    Args:
        X (np.ndarray): Array of shape (n_samples, n_features), representing the first dataset.
        Y (np.ndarray): Array of shape (n_samples, n_features), representing the second dataset.

    Returns:
        float: The mean Wasserstein distance across all feature dimensions.
    """
    distances = [
        wasserstein_distance(X[:, i], Y[:, i])
        for i in range(X.shape[1])
    ]
    return np.mean(distances)

def evaluate_multiclass_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> None:
    """
    Train and evaluate a multiclass Random Forest classifier using stratified K-fold cross-validation.
    Also plots ROC and Precision-Recall curves per class.

    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples,).
        X_test (np.ndarray): Testing feature matrix of shape (n_samples, n_features).
        y_test (np.ndarray): Testing labels of shape (n_samples,).
        n_splits (int, optional): Number of folds for Stratified K-Fold cross-validation. Defaults to 5.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        None

    Outputs:
        - Prints classification report and confusion matrix for each fold.
        - Plots ROC curves per class and the macro-average ROC.
        - Plots Precision-Recall curves per class with Average Precision scores.
        - Prints overall classification metrics across all folds.
    """
    
    target_classes = ['Normal', 'Crackle', 'Wheeze']
    mask_train = np.isin(y_train, target_classes)
    mask_test = np.isin(y_test, target_classes)

    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    classes = np.unique(y_train)
    n_classes = len(classes)
    class_to_idx = {label: i for i, label in enumerate(classes)}

    y_train_bin = np.array([class_to_idx[label] for label in y_train])
    y_test_bin = np.array([class_to_idx[label] for label in y_test])
    y_test_ovr = label_binarize(y_test_bin, classes=list(range(n_classes)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_probs = []
    all_preds = []
    all_trues = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_bin), 1):
        X_tr, y_tr = X_train_scaled[train_idx], y_train_bin[train_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
        clf.fit(X_tr, y_tr)

        y_proba = clf.predict_proba(X_test_scaled)
        y_pred = clf.predict(X_test_scaled)

        all_probs.append(np.array(y_proba))
        all_preds.append(y_pred)
        all_trues.append(y_test_bin)

        print(f"\nFold {fold} Classification Report:")
        print(classification_report(y_test_bin, y_pred, target_names=classes))
        print("Confusion matrix:")
        print(confusion_matrix(y_test_bin, y_pred))

    # ROC Curve per class for each fold
    plt.figure(figsize=(8, 6))
    all_fpr = []
    all_tpr = []
    all_auc = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_idx, probs in enumerate(all_probs, 1):
        y_true_bin = label_binarize(y_test_bin, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tpr.append(interp_tpr)
            all_fpr.append(mean_fpr)
            roc_auc = auc(fpr, tpr)
            all_auc.append(roc_auc)
            plt.plot(fpr, tpr, lw=1.5, label=f"ROC curve of class {classes[i]} in {fold_idx}-fold (area={roc_auc:.2f})")

    mean_tpr = np.mean(all_tpr, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', linestyle='-', linewidth=2,
             label=f"Macro-average ROC (area={macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right", fontsize="x-small")
    plt.tight_layout()
    plt.show()

    # Precision-Recall Curves 
    final_probs = np.concatenate(all_probs)   # shape: (n_samples, n_classes)
    final_trues_bin = label_binarize(np.concatenate(all_trues), classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(final_trues_bin[:, i], final_probs[:, i])
        ap_score = average_precision_score(final_trues_bin[:, i], final_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f"Class {classes[i]} (AP={ap_score:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # Overall Performance
    final_preds = np.concatenate(all_preds)
    final_trues = np.concatenate(all_trues)
    print("\n=== Overall Classification Report ===")
    print(classification_report(final_trues, final_preds, target_names=classes))
