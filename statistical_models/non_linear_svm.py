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

    return grid_acc

def train(X, y, classes, n_splits=5, C=1.0, gamma="scale", class_weight=None):
    """
    Cross-validation training with Calibrated SVM, showing CM + report per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        base_svm = SVC(
            kernel="rbf", 
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        clf = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_percent[i,j]:.1f}%\n{cm[i,j]}"
                
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", 
                    xticklabels=classes, yticklabels=classes, cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.tight_layout()
        plt.show()

        print(classification_report(y_val, y_pred, target_names=classes))
        
    # Final model on full train
    final_model = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight, probability=True, random_state=42)
    calibrated_final = CalibratedClassifierCV(final_model, method="sigmoid", cv=3)
    calibrated_final.fit(X, y)
    return calibrated_final

def evaluate(X_test, y_test, classes, clf):
    """
    Evaluate SVM on test set: CM + report + ROC/PR + metrics.
    """
    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_percent[i,j]:.1f}%\n{cm[i,j]}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.show()

    y_bin = label_binarize(y_test, classes=range(len(classes)))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    plt.figure(figsize=(8, 6))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=class_colors[cls], lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color="black", linestyle="--", lw=2.5, label=f"Macro Avg (AUC={macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=class_colors[cls], lw=2, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    mcm = multilabel_confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    metrics = []
    for i, cls in enumerate(classes):
        tn, fp, fn, tp = mcm[i].ravel()
        SE = tp / (tp + fn + 1e-6)
        SP = tn / (tn + fp + 1e-6)
        PPV = tp / (tp + fp + 1e-6)
        NPV = tn / (tn + fn + 1e-6)
        MCC = (tp * tn - fp * fn) / (np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + 1e-6)
        SCORE = (SE + SP) / 2
        metrics.append({
            "Class": cls,
            "Sensitivity (SE)": SE,
            "Specificity (SP)": SP,
            "PPV": PPV,
            "NPV": NPV,
            "MCC": MCC,
            "SCORE": SCORE
        })
    return pd.DataFrame(metrics)