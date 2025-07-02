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

def evaluate(X_test, y_test, classes, clf):
    """
    Evaluate a trained classifier on test data, print classification report,
    plot confusion matrix, ROC curves (with macro-average), PR curves,
    and return detailed metrics.

    Args:
        X_test (np.ndarray): Test features (samples x features).
        y_test (np.ndarray): Encoded test labels.
        classes (list or np.ndarray): Class names corresponding to label encodings.
        clf (fitted estimator): Trained classifier.

    Returns:
        pd.DataFrame: DataFrame with per-class metrics (SE, SP, PPV, NPV, MCC, SCORE).
    """
    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print("\nClassification Report (Test Set):")
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
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
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.show()

    # Multilabel confusion matrix for per-class metrics
    mcm = multilabel_confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

    # Custom metrics computation
    metrics = []
    for i, label in enumerate(classes):
        tn, fp, fn, tp = mcm[i].ravel()
        SE = tp / (tp + fn + 1e-6)
        SP = tn / (tn + fp + 1e-6)
        PPV = tp / (tp + fp + 1e-6)
        NPV = tn / (tn + fn + 1e-6)
        MCC = (tp * tn - fp * fn) / (np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + 1e-6)
        SCORE = (SE + SP) / 2
        metrics.append({
            "Class": label,
            "Sensitivity (SE)": SE,
            "Specificity (SP)": SP,
            "PPV": PPV,
            "NPV": NPV,
            "MCC": MCC,
            "SCORE": SCORE
        })

    metrics_df = pd.DataFrame(metrics)

    # ROC curves
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=class_colors[class_name], lw=2,
                 label=f"{class_name} (AUC={roc_auc:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color="black", linestyle="--", linewidth=2.5,
             label=f"Macro-average (AUC={macro_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR curves
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=class_colors[class_name], lw=2,
                 label=f"{class_name} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return metrics_df