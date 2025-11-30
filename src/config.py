import os

# Base directory for data
BASE_PATH = "data"

# Processed feature files
AUDIO_FILE = os.path.join(BASE_PATH, "processed", "audio_features_wav2vec2_egemaps.csv")
TEXT_FILE = os.path.join(BASE_PATH, "processed", "text_features_roberta_go_emotions.csv")

# Labels
LABELS_FILE = os.path.join(BASE_PATH, "labels", "detailed_lables.csv")

# Splits
SPLIT_FILES = {
    "train": os.path.join(BASE_PATH, "labels", "train_split.csv"),
    "dev": os.path.join(BASE_PATH, "labels", "dev_split.csv"),
    "test": os.path.join(BASE_PATH, "labels", "test_split.csv")
}

# Number of features to keep after SelectKBest
N_AUDIO_FEATURES = 35
N_TEXT_FEATURES = 35
