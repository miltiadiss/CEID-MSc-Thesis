from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def xgb_grid_search(X, y, param_grid, n_splits=5):
    """
    Perform grid search for XGBoost hyperparameters.

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        y (np.ndarray): Encoded labels.
        param_grid (dict): Hyperparameter grid to search.
        n_splits (int): Number of CV folds.

    Returns:
        GridSearchCV: Fitted grid search object (accuracy).
    """
    base_xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Accuracy grid search
    grid_acc = GridSearchCV(
        base_xgb, param_grid,
        scoring="accuracy",
        cv=skf,
        n_jobs=-1,
        return_train_score=True
    )
    grid_acc.fit(X, y)

    # F1 macro grid search
    grid_f1 = GridSearchCV(
        base_xgb, param_grid,
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
    plt.title("Grid Search XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return grid_acc

def train(X, y, classes, param_dict, n_splits=5):
    """
    Perform cross-validation training with calibrated XGBoost, showing CM + report per fold.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Encoded labels.
        classes (list): Class names.
        param_dict (dict): XGBoost hyperparameters (e.g., n_estimators, max_depth, learning_rate).
        n_splits (int): Number of folds.

    Returns:
        CalibratedClassifierCV: Final trained model on full data.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        base_xgb = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            **param_dict
        )
        clf = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=3)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
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

    # Train final model on full data
    final_xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        **param_dict
    )
    final_clf = CalibratedClassifierCV(final_xgb, method="sigmoid", cv=3)
    final_clf.fit(X, y)

    return final_clf