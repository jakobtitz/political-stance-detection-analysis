# Target-Related Evidence or Dataset Shortcuts?

## Analysing User-Level Political Stance Detection in PolitiSky24

The project investigates whether user-level political stance classifiers base their predictions on meaningful target-related evidence or exploit easier dataset-specific shortcuts. Using the **PolitiSky24** dataset, we compare a lexical **TF-IDF + Logistic Regression** classifier with a fine-tuned **RoBERTa** model for predicting stance toward **Donald Trump** and **Kamala Harris**.

Beyond standard predictive performance, we systematically modify the model inputs to test reliance on the supplied target, explicit candidate mentions, and label-correlated lexical cues. We additionally evaluate how model predictions change when the amount of retrieved user context is reduced.

The three stance classes used throughout the project are **Favor**, **Against**, and **Neither**.

---

## Project Overview

The experimental pipeline consists of six main stages:

1. **Preprocessing**
   Extract and align the PolitiSky24 posting histories, retrieved context posts, stance annotations, and timestamps. Create fixed training, validation, and human-annotated test datasets.

2. **Data analysis**
   Examine dataset composition, label distributions, candidate mentions, available context lengths, and evidence overlap across splits.

3. **Model training**
   Train two models of different complexity:

   * TF-IDF + multinomial logistic regression
   * fine-tuned `roberta-base`

4. **Input intervention construction**
   Create controlled variants of the human-annotated test set by masking or modifying selected information sources.

5. **Input intervention evaluation**
   Evaluate how the frozen TF-IDF and RoBERTa classifiers respond to the interventions.

6. **Context-length analysis**
   Evaluate both models using different numbers of the most recent retrieved context posts.

---

# Installation

## Prerequisites

The project requires:

* **Python 3**
* **Git 2.45.2 or newer**
* **Git LFS**
* the Python packages listed in `requirements.txt`

For local notebook execution, an environment capable of running Jupyter notebooks is also required.

The RoBERTa experiments are computationally more expensive and are intended to be run using a **GPU runtime in Google Colab**.

---

## Clone the Repository

The trained RoBERTa model and several other large files are tracked using **Git LFS**. Git LFS should therefore be installed before cloning the repository.

```bash
git lfs install

git clone https://github.com/jakobtitz/political-stance-detection-analysis.git

cd political-stance-detection-analysis


git lfs pull
```

`git lfs pull` ensures that the actual large files are downloaded rather than only the Git LFS pointer files.

### Git version

Git 2.45.1 introduced a known clone regression affecting repositories that use Git LFS. This can result in an error such as:

```text
fatal: active `post-checkout` hook found during `git clone`
```

The problem is documented in the official Git LFS repository:

[Git LFS issue #5749 — Recent Git security releases break `git clone` on repositories using LFS](https://github.com/git-lfs/git-lfs/issues/5749)

Please use **Git 2.45.2 or newer** when cloning the repository. The installed Git version can be checked with:

```bash
git --version
```

### Clone duration

Cloning the repository can take several minutes because Git LFS has to download the trained RoBERTa model and other relatively large files. During this process, the terminal may remain at the same percentage for some time while an individual LFS object is being downloaded or processed. This does not necessarily indicate that the clone has stalled.

On **Windows PowerShell**, the activity of the Git LFS process can be inspected with:

```powershell
Get-Process git-lfs -ErrorAction SilentlyContinue | Select-Object Id, CPU, WorkingSet
```

Running this command repeatedly while the clone is in progress shows the accumulated CPU time of the `git-lfs` process. If the `CPU` value continues to increase, Git LFS is still actively processing files.

The clone is complete once Git reports:

```text
Filtering content: 100% (...), done.
```

and the normal terminal prompt becomes available again.

---

## Create a Python Environment

Using a virtual environment is recommended.

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

The project is implemented as a sequence of Jupyter notebooks.

For local execution, the notebooks can be opened directly in Visual Studio Code using the Python and Jupyter extensions or through JupyterLab.

If JupyterLab is not already installed, install it separately with:

pip install jupyterlab

Then start JupyterLab from the repository root:

jupyter lab

JupyterLab is intentionally not included in requirements.txt because it is only required for local notebook execution. In particular, Google Colab already provides its own notebook environment, and installing JupyterLab there would add unnecessary notebook-related dependencies.

The notebooks can then be run in the order described below. Each notebook is designed to be executed from top to bottom.

---

# Data

The project is based on the **PolitiSky24** dataset, which contains political Bluesky posts collected around the 2024 U.S. presidential election together with user-level stance annotations.

The original dataset is available through Zenodo:

**PolitiSky24:**
https://zenodo.org/records/15616911

Most raw files required by the project are already included under:

```text
data/raw/
```

However, the following file is approximately 2.1 GB and is therefore **not included in this repository**:

```text
user_post_history_dataset.parquet
```

To reproduce preprocessing from the original data, download the file from:

https://zenodo.org/records/15616911/files/user_post_history_dataset.parquet?download=1

and place it at:

```text
data/raw/user_post_history_dataset.parquet
```

The repository already contains the generated datasets under `data/preprocessed/`. Therefore, downloading the full posting-history file is **only necessary when reproducing the preprocessing pipeline from scratch**.

Model training and the subsequent experiments can be reproduced directly from the provided preprocessed datasets.

---

# Models

Two stance-classification models are included.

## TF-IDF + Logistic Regression

The lexical baseline is stored under:

```text
models/tfidf_logreg/
```

It contains the serialized scikit-learn pipeline and validation metrics produced during training.

The model combines the supplied target and retrieved context posts into one textual representation and classifies it using TF-IDF features and multinomial logistic regression.

## RoBERTa

The fine-tuned RoBERTa model and tokenizer are stored under:

```text
models/roberta/
```

A GPU is strongly recommended for RoBERTa training and evaluation.

---

# Recommended Execution Order

To reproduce the complete pipeline from scratch, run the notebooks in the following order:

```text
01_preprocessing.ipynb
        ↓
02_data_analysis.ipynb
        ↓
03_1_train_tfidf_logreg.ipynb
03_2_train_roberta.ipynb
        ↓
04_1_select_masking_tokens.ipynb
        ↓
04_2_create_input_interventions.ipynb
        ↓
05_1_evaluate_input_interventions_tfidf.ipynb
05_2_evaluate_input_interventions_roberta.ipynb
        ↓
06_1_evaluate_context_length_tfidf.ipynb
06_2_evaluate_context_length_roberta.ipynb
```

The preprocessing outputs, trained models, intervention datasets, and evaluation results are already included in the repository. Individual stages can therefore also be inspected or rerun without necessarily repeating all preceding stages.

---

# Repository Structure

```text
political-stance-detection-analysis│
├── data/
│   ├── raw/
│   │   └── ...                         # Original PolitiSky24 datasets
│   │
│   ├── preprocessed/
│   │   └── ...                         # Preprocessed datasets
│   │
│   └── interventions/
│       └── ...                         # Intervention datasets and audit files
│
├── models/
│   ├── tfidf_logreg/
│   │   ├── model.joblib
│   │   └── validation_class_metrics.csv
│   │
│   └── roberta/
│       └── ...                         # Fine-tuned model, tokenizer and metadata
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_1_train_tfidf_logreg.ipynb
│   ├── 03_2_train_roberta.ipynb
│   ├── 04_1_select_masking_tokens.ipynb
│   ├── 04_2_create_input_interventions.ipynb
│   ├── 05_1_evaluate_input_interventions_tfidf.ipynb
│   ├── 05_2_evaluate_input_interventions_roberta.ipynb
│   ├── 06_1_evaluate_context_length_tfidf.ipynb
│   └── 06_2_evaluate_context_length_roberta.ipynb
│
├── results/
│   ├── tfidf_interventions/
│   │   └── ...                         # TF-IDF intervention evaluation tables
│   │
│   ├── roberta_interventions/
│   │   └── ...                         # RoBERTa intervention evaluation tables
│   │
│   ├── tfidf_context_length/
│   │   └── ...                         # TF-IDF context-length results
│   │
│   └── roberta_context_length/
│       └── ...                         # RoBERTa context-length results
│
├── src/
│   ├── __init__.py
│   └── utils.py                        # Shared evaluation functions
│
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```
