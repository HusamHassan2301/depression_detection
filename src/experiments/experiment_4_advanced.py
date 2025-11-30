# Experiment 4 – Advanced Multimodal Fusion
# (Unimodal ML + Early Fusion ML + Late Fusion ML + Deep Learning Fusion)

# This script is GitHub-ready and Colab-ready.
# It uses fixed Train/Dev/Test splits and produces a full results table.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# PATHS (GitHub-style local paths)
# ================================================================
def load_paths():
    base = Path("data")

    audio = base / "processed" / "audio_features_wav2vec2_egemaps.csv"
    text = base / "processed" / "text_features_distilbert_finetuned.csv"
    labels = base / "labels" / "detailed_lables.csv"

    train = base / "labels" / "train_split.csv"
    dev = base / "labels" / "dev_split.csv"
    test = base / "labels" / "test_split.csv"

    return audio, text, labels, train, dev, test


# ================================================================
# Load CSV files
# ================================================================
def load_split(path):
    df = pd.read_csv(path)
    if "Participant_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Participant_ID"})
    return df["Participant_ID"].astype(int).tolist()


def subset(ids_all, Xa, Xt, y, subset_ids):
    subset_ids = set(subset_ids)
    mask = np.array([pid in subset_ids for pid in ids_all])
    return Xa[mask], Xt[mask], y[mask]


# ================================================================
# Deep Learning Fusion Model
# ================================================================
class FusionNet(nn.Module):
    def __init__(self, a_dim, t_dim, hid=128):
        super().__init__()
        self.audio = nn.Sequential(
            nn.Linear(a_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU()
        )
        self.text = nn.Sequential(
            nn.Linear(t_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Linear(hid//2, 1),
            nn.Sigmoid()
        )

    def forward(self, a, t):
        a = self.audio(a)
        t = self.text(t)
        x = torch.cat([a, t], 1)
        return self.fc(x).squeeze()


class DS(Dataset):
    def __init__(self, xa, xt, y):
        self.xa = torch.FloatTensor(xa)
        self.xt = torch.FloatTensor(xt)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.xa[i], self.xt[i], self.y[i]


# ================================================================
# Experiment 4 main function
# ================================================================
def run_experiment_4():
    print("\n=============================================================")
    print("EXPERIMENT 4: ML (uni + fusion) + Deep Fusion")
    print("=============================================================")

    # -------------------------
    # Load paths and data
    # -------------------------
    A_FILE, T_FILE, L_FILE, TR_FILE, DV_FILE, TE_FILE = load_paths()

    audio = pd.read_csv(A_FILE).rename(columns={pd.read_csv(A_FILE).columns[0]: "id"}).set_index("id")
    text = pd.read_csv(T_FILE).rename(columns={pd.read_csv(T_FILE).columns[0]: "id"}).set_index("id")
    labels = pd.read_csv(L_FILE).rename(columns={pd.read_csv(L_FILE).columns[0]: "id"}).set_index("id")

    ids = audio.index
    Xa = audio.values
    Xt = text.values
    y = labels["Depression_label"].astype(int).values

    tr_ids = load_split(TR_FILE)
    dv_ids = load_split(DV_FILE)
    te_ids = load_split(TE_FILE)

    Xa_tr, Xt_tr, y_tr = subset(ids, Xa, Xt, y, tr_ids)
    Xa_dv, Xt_dv, y_dv = subset(ids, Xa, Xt, y, dv_ids)
    Xa_te, Xt_te, y_te = subset(ids, Xa, Xt, y, te_ids)

    print(f"Train = {len(y_tr)}, Dev = {len(y_dv)}, Test = {len(y_te)}")
    print("Train class distribution:", dict(pd.Series(y_tr).value_counts()))
    print("Test class distribution:", dict(pd.Series(y_te).value_counts()))

    # -------------------------
    # Feature selection + scaling
    # -------------------------
    k_audio = 80
    k_text = 80

    sel_a = SelectKBest(mutual_info_classif, k=k_audio)
    sel_t = SelectKBest(mutual_info_classif, k=k_text)

    Xa_tr2 = sel_a.fit_transform(Xa_tr, y_tr)
    Xa_dv2 = sel_a.transform(Xa_dv)
    Xa_te2 = sel_a.transform(Xa_te)

    Xt_tr2 = sel_t.fit_transform(Xt_tr, y_tr)
    Xt_dv2 = sel_t.transform(Xt_dv)
    Xt_te2 = sel_t.transform(Xt_te)

    sa = StandardScaler()
    st = StandardScaler()

    Xa_tr3 = sa.fit_transform(Xa_tr2)
    Xa_dv3 = sa.transform(Xa_dv2)
    Xa_te3 = sa.transform(Xa_te2)

    Xt_tr3 = st.fit_transform(Xt_tr2)
    Xt_dv3 = st.transform(Xt_dv2)
    Xt_te3 = st.transform(Xt_te2)

    # -------------------------
    # SMOTE only for training
    # -------------------------
    sm = SMOTE(random_state=42)
    Xa_bal, y_bal = sm.fit_resample(Xa_tr3, y_tr)
    Xt_bal, _ = sm.fit_resample(Xt_tr3, y_tr)

    # -------------------------
    # Define ML models
    # -------------------------
    ml_models = [
        ("SVM", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced")),
        ("XGB", XGBClassifier(n_estimators=200, scale_pos_weight=1)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(100, 50)))
    ]

    rows = []

    # -------------------------
    # Unimodal: Audio/Text
    # -------------------------
    for name, model in ml_models:
        model.fit(Xa_bal, y_bal)
        dv_scores = model.predict_proba(Xa_dv3)[:,1]
        thr = precision_recall_curve(y_dv, dv_scores)[2].mean()  # simple threshold

        te_scores = model.predict_proba(Xa_te3)[:,1]
        te_pred = (te_scores > thr).astype(int)

        rows.append(["Audio-only", name, thr,
                     f1_score(y_te, te_pred),
                     precision_score(y_te, te_pred, zero_division=0),
                     recall_score(y_te, te_pred, zero_division=0),
                     roc_auc_score(y_te, te_scores)])

    for name, model in ml_models:
        model.fit(Xt_bal, y_bal)
        dv_scores = model.predict_proba(Xt_dv3)[:,1]
        thr = precision_recall_curve(y_dv, dv_scores)[2].mean()

        te_scores = model.predict_proba(Xt_te3)[:,1]
        te_pred = (te_scores > thr).astype(int)

        rows.append(["Text-only", name, thr,
                     f1_score(y_te, te_pred),
                     precision_score(y_te, te_pred, zero_division=0),
                     recall_score(y_te, te_pred, zero_division=0),
                     roc_auc_score(y_te, te_scores)])

    # -------------------------
    # Early Fusion
    # -------------------------
    Xf_bal = np.hstack([Xa_bal, Xt_bal])
    Xf_dv = np.hstack([Xa_dv3, Xt_dv3])
    Xf_te = np.hstack([Xa_te3, Xt_te3])

    for name, model in ml_models:
        model.fit(Xf_bal, y_bal)
        dv_scores = model.predict_proba(Xf_dv)[:,1]
        thr = precision_recall_curve(y_dv, dv_scores)[2].mean()

        te_scores = model.predict_proba(Xf_te)[:,1]
        te_pred = (te_scores > thr).astype(int)

        rows.append(["Early Fusion", name, thr,
                     f1_score(y_te, te_pred),
                     precision_score(y_te, te_pred, zero_division=0),
                     recall_score(y_te, te_pred, zero_division=0),
                     roc_auc_score(y_te, te_scores)])

    # -------------------------
    # Late Fusion (avg)
    # -------------------------
    for name, model in ml_models:
        # audio
        model.fit(Xa_bal, y_bal)
        dv_a = model.predict_proba(Xa_dv3)[:,1]
        te_a = model.predict_proba(Xa_te3)[:,1]

        # text
        model.fit(Xt_bal, y_bal)
        dv_t = model.predict_proba(Xt_dv3)[:,1]
        te_t = model.predict_proba(Xt_te3)[:,1]

        dv_scores = (dv_a + dv_t) / 2
        thr = precision_recall_curve(y_dv, dv_scores)[2].mean()

        te_scores = (te_a + te_t) / 2
        te_pred = (te_scores > thr).astype(int)

        rows.append(["Late Fusion", name, thr,
                     f1_score(y_te, te_pred),
                     precision_score(y_te, te_pred, zero_division=0),
                     recall_score(y_te, te_pred, zero_division=0),
                     roc_auc_score(y_te, te_scores)])

    # -------------------------
    # Deep Learning Fusion
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FusionNet(Xa_bal.shape[1], Xt_bal.shape[1]).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    train_ds = DS(Xa_bal, Xt_bal, y_bal)
    dv_ds = DS(Xa_dv3, Xt_dv3, y_dv)
    te_ds = DS(Xa_te3, Xt_te3, y_te)

    tldr = DataLoader(train_ds, batch_size=16, shuffle=True)
    dvldr = DataLoader(dv_ds, batch_size=32)
    tel = DataLoader(te_ds, batch_size=32)

    best_f1 = 0
    best_state = None

    for ep in range(60):
        net.train()
        for xa, xt, yy in tldr:
            xa, xt, yy = xa.to(device), xt.to(device), yy.to(device)
            opt.zero_grad()
            out = net(xa, xt)
            l = loss_fn(out, yy)
            l.backward()
            opt.step()

        # dev eval
        net.eval()
        dv_scores = []
        with torch.no_grad():
            for xa, xt, yy in dvldr:
                xa, xt = xa.to(device), xt.to(device)
                dv_scores.extend(net(xa, xt).cpu().numpy())

        dv_scores = np.array(dv_scores)
        dv_thr = precision_recall_curve(y_dv, dv_scores)[2].mean()
        dv_pred = (dv_scores > dv_thr)
        dv_f1 = f1_score(y_dv, dv_pred)

        if dv_f1 > best_f1:
            best_f1 = dv_f1
            best_state = net.state_dict()

    # test with best model
    net.load_state_dict(best_state)
    net.eval()

    te_scores = []
    with torch.no_grad():
        for xa, xt, yy in tel:
            xa, xt = xa.to(device), xt.to(device)
            te_scores.extend(net(xa, xt).cpu().numpy())

    te_scores = np.array(te_scores)
    thr = precision_recall_curve(y_dv, dv_scores)[2].mean()
    te_pred = (te_scores > thr).astype(int)

    rows.append(["Deep Fusion", "FusionNet", thr,
                 f1_score(y_te, te_pred),
                 precision_score(y_te, te_pred, zero_division=0),
                 recall_score(y_te, te_pred, zero_division=0),
                 roc_auc_score(y_te, te_scores)])

    # -------------------------
    # Final Table
    # -------------------------
    df = pd.DataFrame(rows, columns=[
        "Setting", "Model", "Threshold", "F1", "Precision", "Recall", "AUC"
    ])
    print("\n=============================================================")
    print("EXPERIMENT 4 RESULTS TABLE")
    print("=============================================================")
    print(df)

    return df


if __name__ == "__main__":
    run_experiment_4()
