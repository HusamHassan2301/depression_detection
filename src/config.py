from pathlib import Path

# Base directory for your dataset
BASE_PATH = Path("data")

# Processed feature files
AUDIO_FILE = BASE_PATH / "processed" / "audio_features_wav2vec2_egemaps.csv"
TEXT_FILE = BASE_PATH / "processed" / "text_features_roberta_go_emotions.csv"

# Labels
LABELS_FILE = BASE_PATH / "labels" / "detailed_lables.csv"

# Train / Dev / Test split files
SPLIT_FILES = {
    "train": BASE_PATH / "labels" / "train_split.csv",
    "dev": BASE_PATH / "labels" / "dev_split.csv",
    "test": BASE_PATH / "labels" / "test_split.csv",
}

# Feature selection sizes
N_AUDIO_FEATURES = 35
N_TEXT_FEATURES = 35

RANDOM_STATE = 42
