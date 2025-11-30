
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE


AUDIO_FILE = "data/processed/audio_features_wav2vec2_egemaps.csv"
TEXT_FILE = "data/processed/text_features_distilbert_finetuned.csv"
LABELS_FILE = "data/labels/detailed_lables.csv"

TRAIN_SPLIT = "data/labels/train_split.csv"
DEV_SPLIT = "data/labels/dev_split.csv"
TEST_SPLIT = "data/labels/test_split.csv"


def load_split(path):
    df = pd.read_csv(path)
    if "Participant_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Participant_ID"})
    return df["Participant_ID"].astype(int).tolist()


def filter_by_ids(all_ids, X, y, subset_ids):
    subset_ids = set(subset_ids)
    mask = np.array([pid in subset_ids for pid in all_ids])
    return X[mask], y[mask]


def best_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-8)
    if len(t) == 0:
        return 0.5
    return t[np.argmax(f1[:-1])]


def run_experiment_3(k_audio=100, k_text=100):
    audio = pd.read_csv(AUDIO_FILE).rename(columns={pd.read_csv(AUDIO_FILE).columns[0]: "id"}).set_index("id")
    text = pd.read_csv(TEXT_FILE).rename(columns={pd.read_csv(TEXT_FILE).columns[0]: "id"}).set_index("id")
    labels = pd.read_csv(LABELS_FILE).rename(columns={pd.read_csv(LABELS_FILE).columns[0]: "id"}).set_index("id")

    ids = audio.index.intersection(text.index).intersection(labels.index)

    Xa = audio.loc[ids].values
    Xt = text.loc[ids].values
    y = labels.loc[ids, "Depression_label"].astype(int).values

    train_ids = load_split(TRAIN_SPLIT)
    dev_ids = load_split(DEV_SPLIT)
    test_ids = load_split(TEST_SPLIT)

    Xa_tr, y_tr = filter_by_ids(ids, Xa, y, train_ids)
    Xa_dv, y_dv = filter_by_ids(ids, Xa, y, dev_ids)
    Xa_te, y_te = filter_by_ids(ids, Xa, y, test_ids)

    Xt_tr, _ = filter_by_ids(ids, Xt, y, train_ids)
    Xt_dv, _ = filter_by_ids(ids, Xt, y, dev_ids)
    Xt_te, _ = filter_by_ids(ids, Xt, y, test_ids)

    fs_a = SelectKBest(mutual_info_classif, k=min(k_audio, Xa_tr.shape[1]))
    fs_t = SelectKBest(mutual_info_classif, k=min(k_text, Xt_tr.shape[1]))

    Xa_tr = fs_a.fit_transform(Xa_tr, y_tr)
    Xa_dv = fs_a.transform(Xa_dv)
    Xa_te = fs_a.transform(Xa_te)

    Xt_tr = fs_t.fit_transform(Xt_tr, y_tr)
    Xt_dv = fs_t.transform(Xt_dv)
    Xt_te = fs_t.transform(Xt_te)

    sa = StandardScaler()
    st = StandardScaler()

    Xa_tr = sa.fit_transform(Xa_tr)
    Xa_dv = sa.transform(Xa_dv)
    Xa_te = sa.transform(Xa_te)

    Xt_tr = st.fit_transform(Xt_tr)
    Xt_dv = st.transform(Xt_dv)
    Xt_te = st.transform(Xt_te)

    Xf_tr = np.hstack([Xa_tr, Xt_tr])
    Xf_dv = np.hstack([Xa_dv, Xt_dv])
    Xf_te = np.hstack([Xa_te, Xt_te])

    sm = SMOTE(random_state=42)
    Xf_tr_bal, y_tr_bal = sm.fit_resample(Xf_tr, y_tr)

    pos_weight = np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)

    models = {
        "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGB": XGBClassifier(n_estimators=200, scale_pos_weight=pos_weight, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    }

    rows = []

    for name, model in models.items():
        model.fit(Xf_tr_bal, y_tr_bal)

        dv_scores = model.predict_proba(Xf_dv)[:, 1]
        thr = best_threshold(y_dv, dv_scores)

        te_scores = model.predict_proba(Xf_te)[:, 1]
        te_pred = (te_scores > thr).astype(int)

        rows.append({
            "setting": "early_smote_tuned",
            "model": name,
            "threshold": float(thr),
            "f1": f1_score(y_te, te_pred),
            "precision": precision_score(y_te, te_pred, zero_division=0),
            "recall": recall_score(y_te, te_pred, zero_division=0),
            "auc": roc_auc_score(y_te, te_scores),
        })

    out = pd.DataFrame(rows)
    print(out)
    return out


if __name__ == "__main__":
    run_experiment_3()
