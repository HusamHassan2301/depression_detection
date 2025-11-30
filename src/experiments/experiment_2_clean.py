
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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


def run_experiment_2(k_audio=100, k_text=100):
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
    Xf_te = np.hstack([Xa_te, Xt_te])

    models = {
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGB": XGBClassifier(n_estimators=200, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    }

    rows = []

    for name, model in models.items():
        model.fit(Xa_tr, y_tr)
        pa = model.predict_proba(Xa_te)[:, 1]
        pred_a = (pa > 0.5).astype(int)

        model.fit(Xt_tr, y_tr)
        pt = model.predict_proba(Xt_te)[:, 1]
        pred_t = (pt > 0.5).astype(int)

        model.fit(Xf_tr, y_tr)
        pf = model.predict_proba(Xf_te)[:, 1]
        pred_f = (pf > 0.5).astype(int)

        pl = (pa + pt) / 2
        pred_l = (pl > 0.5).astype(int)

        for setting, pred, prob in [
            ("audio", pred_a, pa),
            ("text", pred_t, pt),
            ("early", pred_f, pf),
            ("late", pred_l, pl),
        ]:
            rows.append({
                "setting": setting,
                "model": name,
                "f1": f1_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "auc": roc_auc_score(y_te, prob),
            })

    out = pd.DataFrame(rows)
    print(out)
    return out


if __name__ == "__main__":
    run_experiment_2()
