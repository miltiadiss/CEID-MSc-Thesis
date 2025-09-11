from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, multilabel_confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def knn_grid_search(X, y, n_splits=5, neighbor_range=range(1, 21)):
    """
    Perform grid search on KNN n_neighbors with cross-validation.
    Plots accuracy and F1-score as a function of n_neighbors.

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        y (np.ndarray): Encoded labels (integer codes).
        n_splits (int): Number of CV folds.
        neighbor_range (range): Range of n_neighbors to search.

    Returns:
        None
    """
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(neighbor_range)}

    # Define grid search with accuracy
    grid_search_acc = GridSearchCV(
        knn, param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search_acc.fit(X, y)

    # Define grid search with macro F1
    grid_search_f1 = GridSearchCV(
        knn, param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring=make_scorer(f1_score, average='macro'),
        n_jobs=-1
    )
    grid_search_f1.fit(X, y)

    # Results
    acc_scores = pd.DataFrame(grid_search_acc.cv_results_)
    f1_scores = pd.DataFrame(grid_search_f1.cv_results_)

    results_df = pd.DataFrame({
        "n_neighbors": acc_scores["param_n_neighbors"],
        "mean_accuracy": acc_scores["mean_test_score"],
        "mean_f1_score": f1_scores["mean_test_score"]
    })

    print(f"Best n_neighbors (accuracy): {grid_search_acc.best_params_['n_neighbors']} "
          f"with accuracy: {grid_search_acc.best_score_:.4f}")
    print(f"Best n_neighbors (F1 macro): {grid_search_f1.best_params_['n_neighbors']} "
          f"with F1-score: {grid_search_f1.best_score_:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(results_df["n_neighbors"], results_df["mean_accuracy"], marker='o', label="Accuracy")
    plt.plot(results_df["n_neighbors"], results_df["mean_f1_score"], marker='s', label="F1-score (macro)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Score")
    plt.title("KNN Grid Search")
    plt.legend()
    plt.tight_layout()
    plt.show()

def train(X, y, classes, n_splits=5, n_neighbors=5):
    """
    Perform Stratified K-Fold CV using KNN and plot CM + report for each fold.
    Return the fitted model on the full training set.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        classes (list): Class names.
        n_splits (int): Number of folds.
        n_neighbors (int): Number of neighbors.

    Returns:
        KNeighborsClassifier: Final trained model on full data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        cm = confusion_matrix(y_val, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_percent[i, j]:.1f}%\n{cm[i, j]}"

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=annot, fmt='', cmap="Blues",
                    xticklabels=classes, yticklabels=classes, cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.tight_layout()
        plt.show()

        print(classification_report(y_val, y_pred, target_names=classes))

    # Final model to use for evaluation
    final_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    final_clf.fit(X, y)
    return final_clf