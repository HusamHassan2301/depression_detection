import pandas as pd
import numpy as np

def load_csv(path):
    df = pd.read_csv(path)
    if "Participant_ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Participant_ID"})
    df["Participant_ID"] = df["Participant_ID"].astype(int)
    return df


def get_data_splits(audio_file, text_file, labels_file, split_files):
    # Load full datasets
    audio_df = load_csv(audio_file).set_index("Participant_ID")
    text_df = load_csv(text_file).set_index("Participant_ID")
    labels_df = load_csv(labels_file).set_index("Participant_ID")

    # Only keep IDs present in all modalities
    common = audio_df.index.intersection(text_df.index).intersection(labels_df.index)

    Xa_all = audio_df.loc[common].values
    Xt_all = text_df.loc[common].values
    y_all = labels_df.loc[common, "Depression_label"].astype(int).values

    # Load splits
    splits = {
        name: load_csv(path)["Participant_ID"].tolist()
        for name, path in split_files.items()
    }

    def subset(ids):
        mask = [pid in ids for pid in common]
        return Xa_all[mask], Xt_all[mask], y_all[mask]

    # Build dictionary
    data = {
        "train": {},
        "dev": {},
        "test": {}
    }

    for split in ["train", "dev", "test"]:
        Xa, Xt, y = subset(splits[split])
        data[split]["audio"] = Xa
        data[split]["text"] = Xt
        data[split]["labels"] = y

    return data, common
