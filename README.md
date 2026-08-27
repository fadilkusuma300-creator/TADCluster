# TADCluster

TADCluster is a temporal-adaptive density clustering pipeline for time-aware topic discovery in longitudinal text. It combines semantic distance with a continuous temporal penalty before density clustering, and includes the data preparation, baseline comparison, sensitivity analysis, and evaluation utilities used by the project.

## Project files

- `tadcluster.py` — semantic embedding, UMAP reduction, temporal penalties, semantic-temporal distance, DBSCAN, and HDBSCAN utilities.
- `metrics.py` — c-TF-IDF topic terms, document-level NPMI, Coverage, Temporal Dispersion, B-Cubed metrics, and bootstrap intervals.
- `prepare_data.py` — builders for the Stack Exchange corpora and converter for the News2013 English splits.
- `run_experiments.py` — main comparison, decay-form analysis, matched-granularity analysis, half-life analysis, neighborhood-radius sensitivity, repeated sampling, and News2013 evaluation.
- `config.yaml` — model parameters, search grids, sampling settings, and random seed.
- `requirements.txt` — Python dependencies.

No datasets or generated result files are included in the repository.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The semantic encoder is `sentence-transformers/all-MiniLM-L6-v2`. It is downloaded automatically by Sentence Transformers when first used.

## Directory layout

Create the following directories before preparing the data:

```text
TADCluster/
├── data/
│   ├── raw/
│   │   ├── stackexchange/
│   │   │   ├── stackoverflow.com/
│   │   │   ├── superuser.com/
│   │   │   ├── serverfault.com/
│   │   │   └── askubuntu.com/
│   │   └── news-clustering/
│   ├── D1.csv
│   ├── D2.csv
│   ├── D3.csv
│   ├── news2013_train.csv
│   └── news2013_test.csv
└── results/
```

The prepared D1-D3 tables use:

```text
id,text,timestamp,source
```

The News2013 tables additionally contain:

```text
event_label
```

Timestamps are stored as UTC datetimes.

## Dataset download and preparation

### Stack Exchange

D1 and D3 use Stack Overflow. D2 uses SuperUser, ServerFault, and AskUbuntu.

Public data dump:

- https://archive.org/details/stackexchange

Stack Exchange Data Explorer source repository:

- https://github.com/StackExchange/StackExchange.DataExplorer

Download the required archives from the public data dump and extract them so the raw directory contains:

```text
data/raw/stackexchange/stackoverflow.com/Posts.xml
data/raw/stackexchange/stackoverflow.com/Comments.xml
data/raw/stackexchange/superuser.com/Posts.xml
data/raw/stackexchange/serverfault.com/Posts.xml
data/raw/stackexchange/askubuntu.com/Posts.xml
```

For Stack Overflow, the dump provides `stackoverflow.com-Posts.7z` and `stackoverflow.com-Comments.7z` separately. The other three sites are distributed as site archives containing `Posts.xml` and the other public tables.

Build D1:

```bash
python prepare_data.py d1 \
  --stackoverflow data/raw/stackexchange/stackoverflow.com \
  --output data/D1.csv
```

D1 uses Stack Overflow questions created from `2021-01-01T00:00:00Z` inclusive to `2023-01-01T00:00:00Z` exclusive, score at least 5, and at least three comments. Sampling quotas are Java 425, Python 425, JavaScript 424, and machine-learning 424.

Build D2:

```bash
python prepare_data.py d2 \
  --superuser data/raw/stackexchange/superuser.com \
  --serverfault data/raw/stackexchange/serverfault.com \
  --askubuntu data/raw/stackexchange/askubuntu.com \
  --start 2023-04-01T00:00:00Z \
  --stop 2023-10-01T00:00:00Z \
  --output data/D2.csv
```

D2 samples 300 questions from each site within the stated UTC interval.

Build D3:

```bash
python prepare_data.py d3 \
  --stackoverflow data/raw/stackexchange/stackoverflow.com \
  --output data/D3.csv
```

D3 uses year-stratified Stack Overflow sampling: 3,333 questions from each year from 2017 through 2020 and 3,332 questions from each of 2021 and 2022.

### News2013

The News2013 files are distributed through the Priberam public repository:

- https://github.com/Priberam/news-clustering

Clone the repository under `data/raw/`:

```bash
git clone https://github.com/Priberam/news-clustering.git data/raw/news-clustering
cd data/raw/news-clustering
bash download_data.sh
cd ../../..
```

The download script places `dataset.dev.json` and `dataset.test.json` under the repository's `dataset/` directory. Convert the English splits with:

```bash
python prepare_data.py news2013 \
  --dev data/raw/news-clustering/dataset/dataset.dev.json \
  --test data/raw/news-clustering/dataset/dataset.test.json \
  --output-dir data
```

The converter checks for 12,233 English development documents in 593 events and 8,726 English test documents in 222 events.

## Upstream libraries

- Sentence Transformers: https://github.com/huggingface/sentence-transformers
- UMAP: https://github.com/lmcinnes/umap
- HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan
- BERTopic: https://github.com/MaartenGr/BERTopic
- FASTopic: https://github.com/bobxwu/FASTopic
- TopMost: https://github.com/bobxwu/TopMost

## Running TADCluster

Run the complete analysis pipeline:

```bash
python run_experiments.py \
  --config config.yaml \
  --data-dir data \
  --output-dir results \
  --only all
```

Available analysis targets are:

```text
main
ablation
matched
half-life
epsilon
repeated
news2013
all
```

Example:

```bash
python run_experiments.py \
  --config config.yaml \
  --data-dir data \
  --output-dir results \
  --only main
```

Generated JSON files are written to `results/`.

## Parameter selection

For D1-D3, each corpus is divided with a temporally stratified 20% calibration subset and an 80% evaluation subset. TADCluster selects `alpha` by calibration NPMI. BERTopic-HDBSCAN and FASTopic use the same calibration subset for their parameter grids. The evaluation subset is not used to rank candidate configurations.

The matched-granularity analysis selects HDBSCAN settings by proximity in cluster count and Coverage. NPMI is calculated after the matching configuration has been selected.

For News2013, HDBSCAN parameters and `alpha` are selected on the development split using unlabeled NPMI. Event labels are used for B-Cubed scoring on the test split after the clustering configuration has been fixed.

Randomized operations use seed `42` unless repeated sampling changes the seed for the corresponding run.

## Computational requirements

The semantic-temporal distance matrix is dense and requires quadratic memory in the number of documents. D3 therefore requires several gigabytes of working memory in addition to the embedding and clustering libraries. Pairwise distances are stored as `float32` where possible and converted to double precision only for the HDBSCAN backend.
