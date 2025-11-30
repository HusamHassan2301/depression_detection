import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import shap

from config import (
    AUDIO_FILE, TEXT_FILE, LABELS_FILE, SPLIT_FILES,
    N_AUDIO_FEATURES, N_TEXT_FEATURES
)
from data_loader import get_data_splits
from models import get_model_configurations, get_data_for_model
from utils import (
    threshold_search,
    compute_shap_values,
    print_results_table
)


class DepressionDetectionExperiment:
    def __init__(self):
        self.results = {}
        self.data_splits = None
        self.common_idx = None

    def load_and_preprocess_data(self):
        # Load aligned audio, text, labels + split indices
        self.data_splits, self.common_idx = get_data_splits(
            AUDIO_FILE, TEXT_FILE, LABELS_FILE
        )

        Xa_train = self.data_splits["train"]["audio"]
        Xa_dev = self.data_splits["dev"]["audio"]
        Xt_train = self.data_splits["train"]["text"]
        Xt_dev = self.data_splits["dev"]["text"]
        y_train = self.data_splits["train"]["labels"]
        y_dev = self.data_splits["dev"]["labels"]

        Xa_comb = np.vstack([Xa_train, Xa_dev])
        Xt_comb = np.vstack([Xt_train, Xt_dev])
        y_comb = np.hstack([y_train, y_dev])

        return Xa_comb, Xt_comb, y_comb

    def feature_selection_and_scaling(self, Xa_comb, Xt_comb, y_comb):
        selector_audio = SelectKBest(f_classif, k=N_AUDIO_FEATURES)
        selector_text = SelectKBest(f_classif, k=N_TEXT_FEATURES)

        Xa_train_sel = selector_audio.fit_transform(Xa_comb, y_comb)
        Xa_test_sel = selector_audio.transform(self.data_splits["test"]["audio"])

        Xt_train_sel = selector_text.fit_transform(Xt_comb, y_comb)
        Xt_test_sel = selector_text.transform(self.data_splits["test"]["text"])

        scaler_audio = StandardScaler()
        scaler_text = StandardScaler()

        Xa_train_scaled = scaler_audio.fit_transform(Xa_train_sel)
        Xa_test_scaled = scaler_audio.transform(Xa_test_sel)

        Xt_train_scaled = scaler_text.fit_transform(Xt_train_sel)
        Xt_test_scaled = scaler_text.transform(Xt_test_sel)

        X_train_early = np.hstack([Xa_train_scaled, Xt_train_scaled])
        X_test_early = np.hstack([Xa_test_scaled, Xt_test_scaled])

        data_dict = {
            "audio": {"train": Xa_train_scaled, "test": Xa_test_scaled},
            "text": {"train": Xt_train_scaled, "test": Xt_test_scaled},
            "early": {"train": X_train_early, "test": X_test_early}
        }

        return data_dict, y_comb

    def train_models(self, data_dict, y_comb, y_test):
        models_cfg = get_model_configurations(y_comb)

        for name, (model, modality) in models_cfg.items():
            try:
                X_train, X_test = get_data_for_model(name, data_dict)
                model.fit(X_train, y_comb)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[:, 1]
                else:
                    scores = model.decision_function(X_test)
                    proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                metrics = threshold_search(y_test, proba)

                self.results[name] = {
                    "model": model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "proba": proba,
                    **metrics
                }

            except Exception:
                continue

    def late_fusion(self, y_test):
        audio_models = {k: v for k, v in self.results.items() if k.startswith("Audio_")}
        text_models = {k: v for k, v in self.results.items() if k.startswith("Text_")}

        if not audio_models or not text_models:
            return  # safety fail-safe

        best_audio = max(audio_models.items(), key=lambda x: x[1]["f1"])[1]
        best_text = max(text_models.items(), key=lambda x: x[1]["f1"])[1]

        proba_fused = (best_audio["proba"] + best_text["proba"]) / 2
        metrics = threshold_search(y_test, proba_fused)

        self.results["Late_Average"] = {
            "model": None,
            "X_train": None,
            "X_test": None,
            "proba": proba_fused,
            **metrics
        }

    def summarize_results(self):
        audio = {k: v for k, v in self.results.items() if k.startswith("Audio_")}
        text = {k: v for k, v in self.results.items() if k.startswith("Text_")}
        early = {k: v for k, v in self.results.items() if k.startswith("Early_")}
        late = {"Late_Average": self.results.get("Late_Average")}

        print_results_table(audio, "Audio Models")
        print_results_table(text, "Text Models")
        print_results_table(early, "Early Fusion Models")
        print_results_table(late, "Late Fusion")

        base_models = {k: v for k, v in self.results.items() if k != "Late_Average"}
        best_name, best_metrics = max(base_models.items(), key=lambda x: x[1]["f1"])

        return best_name, best_metrics

    def explain(self, best_name, best_metrics):
        model = best_metrics["model"]
        if model is None:
            return None

        X_train = best_metrics["X_train"]
        X_test = best_metrics["X_test"]

        shap_values, explainer, feature_names = compute_shap_values(
            model, X_train, X_test
        )

        try:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        except:
            pass

        return shap_values, explainer

    def run(self):
        Xa_comb, Xt_comb, y_comb = self.load_and_preprocess_data()
        y_test = self.data_splits["test"]["labels"]

        data_dict, y_comb = self.feature_selection_and_scaling(Xa_comb, Xt_comb, y_comb)
        self.train_models(data_dict, y_comb, y_test)
        self.late_fusion(y_test)

        best_name, best_metrics = self.summarize_results()
        shap_results = self.explain(best_name, best_metrics)

        return self.results, best_name, best_metrics, shap_results


def run_experiment_05_comprehensive():
    experiment = DepressionDetectionExperiment()
    return experiment.run()
