import pandas as pd
import numpy as np
from config import AUDIO_FILE, TEXT_FILE, LABELS_FILE, SPLIT_FILES


def _load_split(path):
    """Read split CSV and return participant IDs."""
    df = pd.read_csv(path)
    if "Participant_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Participant_ID"})
    return df["Participant_ID"].astype(int).tolist()


def _subset_by_ids(all_ids, Xa, Xt, y, subset_ids):
    """Select samples based on participant IDs."""
    subset_ids = set(subset_ids)
    mask = np.array([pid in subset_ids for pid in all_ids])
    return Xa[mask], Xt[mask], y[mask]


def get_data_splits(audio_file=AUDIO_FILE,
                    text_file=TEXT_FILE,
                    labels_file=LABELS_FILE):
    """Load audio, text and labels, align on Participant_ID, return split dictionaries."""

    # Read data
    audio = pd.read_csv(audio_file)
    text = pd.read_csv(text_file)
    labels = pd.read_csv(labels_file)

    # Standardize indexing
    audio = audio.rename(columns={audio.columns[0]: "Participant_ID"}).set_index("Participant_ID")
    text = text.rename(columns={text.columns[0]: "Participant_ID"}).set_index("Participant_ID")
    labels = labels.rename(columns={labels.columns[0]: "Participant_ID"}).set_index("Participant_ID")

    # Find participants present in all files
    common_ids = audio.index.intersection(text.index).intersection(labels.index)

    # Extract aligned arrays
    Xa_all = audio.loc[common_ids].values
    Xt_all = text.loc[common_ids].values
    y_all = labels.loc[common_ids, "Depression_label"].astype(int).values

    # Load split IDs
    train_ids = _load_split(SPLIT_FILES["train"])
    dev_ids = _load_split(SPLIT_FILES["dev"])
    test_ids = _load_split(SPLIT_FILES["test"])

    # Subset arrays
    Xa_train, Xt_train, y_train = _subset_by_ids(common_ids, Xa_all, Xt_all, y_all, train_ids)
    Xa_dev, Xt_dev, y_dev = _subset_by_ids(common_ids, Xa_all, Xt_all, y_all, dev_ids)
    Xa_test, Xt_test, y_test = _subset_by_ids(common_ids, Xa_all, Xt_all, y_all, test_ids)

    # Return full structure
    return {
        "train": {"audio": Xa_train, "text": Xt_train, "labels": y_train},
        "dev":   {"audio": Xa_dev,   "text": Xt_dev,   "labels": y_dev},
        "test":  {"audio": Xa_test,  "text": Xt_test,  "labels": y_test},
    }, common_ids
