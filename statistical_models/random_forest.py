import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report, multilabel_confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.metrics import f1_score, make_scorer

def rf_grid_search(X, y, param_grid, n_splits=5):
    """
    Perform grid search for Random Forest hyperparameters (accuracy + F1 macro).
    Plots point for each hyperparameter combination (accuracy and F1).

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        param_grid (dict): Parameters to search.
        n_splits (int): Number of CV folds.

    Returns:
        GridSearchCV: Fitted accuracy grid search object.
    """
    rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Accuracy grid search
    grid_acc = GridSearchCV(
        rf, param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        return_train_score=True
    )
    grid_acc.fit(X, y)

    # F1 macro grid search
    grid_f1 = GridSearchCV(
        rf, param_grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=skf,
        n_jobs=-1
    )
    grid_f1.fit(X, y)

    # Collect results
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
    plt.title("Grid Search Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return grid_acc

def train(X, y, classes, best_params, n_splits=5):
    """
    Perform cross-validation training with calibrated RF, showing CM + report per fold.
    Fit final model on all training data using best_params and return it.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        classes (list): Class names.
        best_params (dict): Best hyperparameters from grid search.
        n_splits (int): Number of CV folds.

    Returns:
        CalibratedClassifierCV: Final fitted model.
    """
    class_colors = {
        "Normal": "#8ecae6",
        "Crackle": "#fb8500",
        "Wheeze": "#ff006e"
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        base_rf = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            **best_params
        )
        clf = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        cm = confusion_matrix(y_val, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_percent[i,j]:.1f}%\n{cm[i,j]}"

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=annot, fmt='', cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

        print(classification_report(y_val, y_pred, target_names=classes))

    # Fit final model on all data
    final_base_rf = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=42,
        **best_params
    )
    final_clf = CalibratedClassifierCV(final_base_rf, method="sigmoid", cv=3)
    final_clf.fit(X, y)
    return final_clf

def evaluate(X_test, y_test, classes, clf):
    """
    Evaluate RF classifier on test set: CM, classification report, ROC/PR curves, metrics df.

    Args:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        classes (list): Class names.
        clf: Trained classifier.

    Returns:
        pd.DataFrame: Metrics summary per class.
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
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_percent[i,j]:.1f}%\n{cm[i,j]}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC
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
    plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', linewidth=2.5,
             label=f"Macro Avg (AUC={macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'gray', linestyle=':')
    plt.title("ROC Curves (Test Set)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=class_colors[cls], lw=2,
                 label=f"{cls} (AP={ap:.2f})")
    plt.title("Precision-Recall Curves (Test Set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Metrics
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
