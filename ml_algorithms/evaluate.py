import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report, multilabel_confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_model(X_test, y_test, classes, clf):
    """
    Evaluate the classifier on test set: compute Confusion Matrix, Classification Report, ROC/PR curves and metrics (SE, SP, SCORE, PPV, NPV, MCC).

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