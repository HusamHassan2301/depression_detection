from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_model_configurations(y_combined):
    pos_w = (y_combined == 0).sum() / (y_combined == 1).sum()

    xgb_params = {
        "n_estimators": 120,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "scale_pos_weight": pos_w,
        "random_state": 42,
    }

    models = {
        # Audio
        "Audio_LogReg": (LogisticRegression(class_weight="balanced", max_iter=2000), "audio"),
        "Audio_RF": (RandomForestClassifier(class_weight="balanced", n_estimators=200), "audio"),
        "Audio_XGB": (XGBClassifier(**xgb_params), "audio"),
        "Audio_SVM": (SVC(kernel="rbf", class_weight="balanced", probability=True), "audio"),
        "Audio_MLP": (MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500), "audio"),

        # Text
        "Text_LogReg": (LogisticRegression(class_weight="balanced", max_iter=2000), "text"),
        "Text_RF": (RandomForestClassifier(class_weight="balanced", n_estimators=200), "text"),
        "Text_XGB": (XGBClassifier(**xgb_params), "text"),
        "Text_SVM": (SVC(kernel="rbf", class_weight="balanced", probability=True), "text"),
        "Text_MLP": (MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500), "text"),

        # Early fusion
        "Early_LogReg": (LogisticRegression(class_weight="balanced", max_iter=2000), "early"),
        "Early_RF": (RandomForestClassifier(class_weight="balanced", n_estimators=200), "early"),
        "Early_XGB": (XGBClassifier(**xgb_params), "early"),
        "Early_SVM": (SVC(kernel="rbf", class_weight="balanced", probability=True), "early"),
        "Early_MLP": (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500), "early"),
    }

    return models


def get_data_for_model(name, data_dict):
    if name.startswith("Audio_"):
        return data_dict["audio"]["train"], data_dict["audio"]["test"]
    elif name.startswith("Text_"):
        return data_dict["text"]["train"], data_dict["text"]["test"]
    else:
        return data_dict["early"]["train"], data_dict["early"]["test"]
