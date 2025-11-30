import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import shap


def threshold_search(y_true, proba, thresholds=np.arange(0.3, 0.8, 0.02)):
    best = {"f1": 0, "precision": 0, "recall": 0, "accuracy": 0, "threshold": 0.5}

    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best["f1"]:
            best["f1"] = f1
            best["precision"] = precision_score(y_true, y_pred)
            best["recall"] = recall_score(y_true, y_pred)
            best["accuracy"] = accuracy_score(y_true, y_pred)
            best["threshold"] = float(th)

    return best


def print_results_table(results, title):
    print("\n" + title)
    print("Model              | F1     | Precision | Recall | Accuracy | Thr")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:<18} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['accuracy']:.4f} | {m['threshold']:.2f}")


def analyze_model_families(results):
    families = {
        "LogReg": {},
        "RF": {},
        "XGB": {},
        "SVM": {},
        "MLP": {},
    }
    for name, metrics in results.items():
        if "LogReg" in name:
            families["LogReg"][name] = metrics
        elif "RF" in name:
            families["RF"][name] = metrics
        elif "XGB" in name:
            families["XGB"][name] = metrics
        elif "SVM" in name:
            families["SVM"][name] = metrics
        elif "MLP" in name:
            families["MLP"][name] = metrics
    return families


def compute_shap_values(model, X_train, X_test):
    feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    if hasattr(model, "predict_proba"):
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1],
                                         shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test, nsamples=100)
    else:
        explainer = shap.KernelExplainer(lambda x: model.decision_function(x),
                                         shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test, nsamples=100)

    return shap_values[0], explainer, feature_names
