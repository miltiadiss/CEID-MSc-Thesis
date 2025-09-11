import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    make_scorer, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

def grid_search_svm(X, y, param_grid, n_splits=5):
    """
    Perform grid search for SVM (RBF kernel) hyperparameters.

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        y (np.ndarray): Encoded labels.
        param_grid (dict): Hyperparameter grid to search.
        n_splits (int): Number of CV folds.

    Returns:
        GridSearchCV: Fitted grid search object (accuracy).
    """
    base_svc = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Accuracy grid search
    grid_acc = GridSearchCV(
        base_svc, param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        return_train_score=True
    )
    grid_acc.fit(X, y)

    # F1 macro grid search
    grid_f1 = GridSearchCV(
        base_svc, param_grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=skf,
        n_jobs=-1
    )
    grid_f1.fit(X, y)

    # Results
    results_df = pd.DataFrame({
        "params": grid_acc.cv_results_["params"],
        "mean_accuracy": grid_acc.cv_results_["mean_test_score"],
        "mean_f1_macro": grid_f1.cv_results_["mean_test_score"]
    })

    print(f"Best params (accuracy): {grid_acc.best_params_} acc={grid_acc.best_score_:.4f}")
    print(f"Best params (F1 macro): {grid_f1.best_params_} f1={grid_f1.best_score_:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    for i, p in enumerate(results_df["params"]):
        plt.scatter(i, results_df["mean_accuracy"][i], color="blue", label="Accuracy" if i == 0 else "")
        plt.scatter(i, results_df["mean_f1_macro"][i], color="red", label="F1 macro" if i == 0 else "")
    plt.xticks(range(len(results_df)), [str(p) for p in results_df["params"]], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Grid Search SVM (RBF kernel)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def train(X, y, classes, best_params, n_splits=5):
    """Cross-validated training with a Calibrated SVM using provided best hyperparameters.

    Performs Stratified K-Fold cross-validation, plots a confusion matrix (counts + row‑wise %)
    and prints a classification report per fold. Finally, fits a calibrated SVM on the full
    dataset using the same hyperparameters and returns it.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
        classes (List[str]): Ordered list of class names for reporting/plots.
        best_params (Dict): Best hyperparameters for `sklearn.svm.SVC` (e.g., {"C": 2.0, "gamma": 0.1, "kernel": "rbf", "class_weight": "balanced"}).
        n_splits (int, optional): Number of CV folds. Defaults to 5.

    Returns:
        CalibratedClassifierCV: Final fitted calibrated SVM on the full dataset.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Base SVM with provided best params
        base_svm = SVC(
            random_state=42,
            **best_params
        )
        # Calibrate probabilities (Platt scaling by default with method="sigmoid")
        clf = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        # Confusion matrix with row-wise percentages
        cm = confusion_matrix(y_val, y_pred, labels=np.arange(len(classes)))
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_percent[i, j]:.1f}%\n{cm[i, j]}"

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.tight_layout()
        plt.show()

        print(classification_report(y_val, y_pred, target_names=classes))

    # Fit final calibrated model on all data
    final_base = SVC(random_state=42, **best_params)
    final_clf = CalibratedClassifierCV(final_base, method="sigmoid", cv=3)
    final_clf.fit(X, y)
    return final_clf
