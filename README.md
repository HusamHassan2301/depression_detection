Multimodal Depression Detection – Experiment Framework

This repository provides a reproducible framework for multimodal depression-detection experiments using audio and text features extracted from the E-DAIC dataset. The project follows a modular architecture suitable for academic research, supporting transparency, clarity, and future extensibility.

Overview

This work implements a multimodal machine-learning pipeline to evaluate predictive patterns derived from structured audio features (wav2vec2 and eGeMAPS) and text-based semantic–emotional representations (RoBERTa GoEmotions). The framework supports unimodal learning, early-fusion, late-fusion, and multilayer perceptron (MLP) models. It also integrates SHAP-based interpretability to analyse feature contributions and model behaviour.

The primary objective is to examine whether combining modalities provides measurable improvements over unimodal baselines in the context of automated depression screening.

Data Availability

The E-DAIC dataset used in this research is distributed under licence and cannot be included in this repository.
Users must obtain the dataset directly from the authorised provider and adhere to its usage restrictions.

Repository Structure

project_root/
├── README.md
├── requirements.txt
├── config/
│   └── config.py
├── data_processing/
│   └── data_loader.py
├── models/
│   └── models.py
├── utils/
│   └── utils.py
└── experiments/
    ├── experiment_1_baseline.py
    ├── experiment_2_clean.py
    ├── experiment_3_smote_tuned.py
    ├── experiment_4_advanced.py
    └── experiment_05.py

This structure separates configuration files, data-loading modules, modelling functions, utilities, and experiment scripts to facilitate reproducible and systematic experimentation.

Data Requirements

Once users obtain access to the dataset, the files should be arranged as follows:

data/
├── processed/
│   ├── audio_features_wav2vec2_egemaps.csv
│   └── text_features_roberta_go_emotions.csv
└── labels/
    ├── detailed_lables.csv
    ├── train_split.csv
    ├── dev_split.csv
    └── test_split.csv

Each file must include a Participant_ID column to enable alignment between modalities.

Running Experiment 05

Python usage:

    from experiments.experiment_05 import run_experiment_05_comprehensive
    results, best_model_name, best_metrics, shap_outputs = run_experiment_05_comprehensive()

Terminal usage:

    python experiments/experiment_05.py

Summary of Experiment 05

Experiment 05 evaluates the following model families across audio-only, text-only, and fusion configurations:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine
- Multilayer Perceptron (MLP)

Evaluation includes F1-score, Precision, Recall, Accuracy, and threshold optimisation.

Key Findings (final run)

- Best unimodal model: Text Logistic Regression (F1 ≈ 0.5306)
- Best fusion model: Early-Fusion Logistic Regression (F1 ≈ 0.5667)
- Late-fusion result: F1 ≈ 0.5614
- Fusion gain: Early fusion improved F1 compared with both unimodal baselines
- Interpretability: SHAP analyses showed distinct feature-importance patterns between linear and non-linear models, indicating complementary modality contributions

Requirements

Install project dependencies:

    pip install -r requirements.txt

Intended Use

This repository is suitable for:

- Academic dissertations
- Machine-learning research
- Multimodal modelling studies
- Reproducible experimentation pipelines
- Extension to new architectures or datasets

Example Results Summary

- Best unimodal text model: F1 ≈ 0.53
- Best early-fusion model: F1 ≈ 0.56
- Late fusion also demonstrated a measurable improvement
- SHAP analyses highlighted influential emotional-valence text features and spectral–prosodic audio markers

Citation

Husam Hassan (2025). Multimodal Machine-Learning Framework for Depression Detection. Unpublished manuscript.
