# TADCluster

## Description

TADCluster is a temporal-adaptive density clustering pipeline for time-aware topic discovery in longitudinal text. It combines semantic distance with a continuous temporal penalty before density clustering so that local neighborhoods reflect both content similarity and temporal proximity. The repository contains dataset preparation, clustering, baseline comparison, sensitivity analysis, and evaluation utilities.

Repository: https://github.com/fadilkusuma300-creator/TADCluster

## Dataset Information

The project uses three Stack Exchange corpora and the News2013 event-clustering benchmark. Raw datasets are not stored in this repository.

| Dataset | Source | Construction used by the project |
| --- | --- | --- |
| D1 | Stack Overflow public data dump | 1,698 questions from 2021-01-01 to 2023-01-01; score >= 5; at least three comments; tag quotas: Java 425, Python 425, JavaScript 424, machine-learning 424 |
| D2 | SuperUser, ServerFault, AskUbuntu public data dumps | 300 questions per site from 2023-04-01 to 2023-10-01 UTC |
| D3 | Stack Overflow public data dump | Year-stratified sample: 3,333 questions per year for 2017-2020 and 3,332 per year for 2021-2022 |
| News2013 | Priberam news-clustering repository | English development split: 12,233 documents / 593 events; test split: 8,726 documents / 222 events |

Public sources:

- Stack Exchange data dump: https://archive.org/details/stackexchange
- Stack Exchange Data Explorer source repository: https://github.com/StackExchange/StackExchange.DataExplorer
- News2013: https://github.com/Priberam/news-clustering

Prepared D1-D3 tables use:

```text
id,text,timestamp,source
```

News2013 tables additionally contain:

```text
event_label
```

Timestamps are stored as UTC datetimes.

## Code Information

- `tadcluster.py` - sentence embedding, UMAP reduction, temporal penalties, semantic-temporal distance construction, DBSCAN, and HDBSCAN utilities.
- `metrics.py` - c-TF-IDF topic terms, document-level NPMI, Coverage, Temporal Dispersion, B-Cubed metrics, and bootstrap intervals.
- `prepare_data.py` - builders for D1-D3 and conversion of the News2013 English splits.
- `run_experiments.py` - main comparison, decay-form analysis, matched-granularity analysis, half-life analysis, neighborhood-radius sensitivity, repeated sampling, and News2013 evaluation.
- `config.yaml` - model parameters, search grids, sampling settings, and random seed.
- `requirements.txt` - Python dependencies.

No datasets, generated result files, notebooks, or test scripts are included in the repository.

## Requirements

A Linux environment is recommended. The computing environment used for the reported analyses is:

- Ubuntu 22.04 LTS
- Python 3.11
- CUDA 12.1
- Intel Xeon-class CPU
- 128 GB system memory
- NVIDIA RTX A6000 GPU with 48 GB VRAM

Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The semantic encoder is `sentence-transformers/all-MiniLM-L6-v2` and is downloaded automatically by Sentence Transformers when first used. Exact direct dependency pins are listed in `requirements.txt`.

The semantic-temporal distance matrix is dense and has quadratic memory growth in the number of documents. D3 therefore requires substantial system memory in addition to the embedding and clustering libraries. Pairwise distances are stored as `float32` where possible and converted to double precision only for the HDBSCAN backend.

## Directory Layout

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

## Usage Instructions

### 1. Download the Stack Exchange data

Download the required archives from:

https://archive.org/details/stackexchange

Extract the files so that the raw-data directory contains:

```text
data/raw/stackexchange/stackoverflow.com/Posts.xml
data/raw/stackexchange/stackoverflow.com/Comments.xml
data/raw/stackexchange/superuser.com/Posts.xml
data/raw/stackexchange/serverfault.com/Posts.xml
data/raw/stackexchange/askubuntu.com/Posts.xml
```

For Stack Overflow, the public dump provides the posts and comments archives separately. The other three sites are distributed as site archives containing `Posts.xml` and other public tables.

### 2. Build D1

```bash
python prepare_data.py d1 \
  --stackoverflow data/raw/stackexchange/stackoverflow.com \
  --output data/D1.csv
```

### 3. Build D2

```bash
python prepare_data.py d2 \
  --superuser data/raw/stackexchange/superuser.com \
  --serverfault data/raw/stackexchange/serverfault.com \
  --askubuntu data/raw/stackexchange/askubuntu.com \
  --start 2023-04-01T00:00:00Z \
  --stop 2023-10-01T00:00:00Z \
  --output data/D2.csv
```

### 4. Build D3

```bash
python prepare_data.py d3 \
  --stackoverflow data/raw/stackexchange/stackoverflow.com \
  --output data/D3.csv
```

### 5. Download and convert News2013

```bash
git clone https://github.com/Priberam/news-clustering.git data/raw/news-clustering
cd data/raw/news-clustering
bash download_data.sh
cd ../../..

python prepare_data.py news2013 \
  --dev data/raw/news-clustering/dataset/dataset.dev.json \
  --test data/raw/news-clustering/dataset/dataset.test.json \
  --output-dir data
```

The converter checks the expected English development and test split sizes before writing the prepared tables.

### 6. Run the analysis pipeline

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

For a single target, for example the main comparison:

```bash
python run_experiments.py \
  --config config.yaml \
  --data-dir data \
  --output-dir results \
  --only main
```

Generated JSON files are written to `results/`.

## Methodology

The analysis pipeline follows these steps:

1. Clean the public text records while retaining technical identifiers, technology names, and timestamps.
2. Encode documents with `sentence-transformers/all-MiniLM-L6-v2`.
3. Reduce the sentence embeddings with UMAP using the settings in `config.yaml`.
4. Compute normalized semantic distances and pairwise time gaps.
5. Convert time gaps to a continuous temporal penalty controlled by a corpus-level half-life.
6. Combine semantic and temporal terms to form the semantic-temporal distance matrix.
7. Apply density clustering using the precomputed distance matrix.
8. Extract cluster terms with c-TF-IDF and evaluate semantic coherence, Coverage, and Temporal Dispersion.
9. Use the News2013 development split for unlabeled model selection and the test split for B-Cubed evaluation against event labels.
10. Run the configured decay-form, granularity, half-life, neighborhood-radius, and repeated-sampling analyses as required.

## Parameter Selection

For D1-D3, each corpus is divided with a temporally stratified 20% calibration subset and an 80% evaluation subset. TADCluster selects `alpha` by calibration NPMI. BERTopic-HDBSCAN and FASTopic use the same calibration subset for their parameter grids. The evaluation subset is not used to rank candidate configurations.

The matched-granularity analysis selects HDBSCAN settings by proximity in cluster count and Coverage. NPMI is calculated after the matching configuration has been selected.

For News2013, HDBSCAN parameters and `alpha` are selected on the development split using unlabeled NPMI. Event labels are used for B-Cubed scoring on the test split after the clustering configuration has been fixed.

Randomized operations use seed `42` unless repeated sampling changes the seed for the corresponding run.

## Upstream Libraries

- Sentence Transformers: https://github.com/huggingface/sentence-transformers
- UMAP: https://github.com/lmcinnes/umap
- HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan
- BERTopic: https://github.com/MaartenGr/BERTopic
- FASTopic: https://github.com/bobxwu/FASTopic
- TopMost: https://github.com/bobxwu/TopMost

## Citations

If this project is used in scholarly work, please cite the associated study:

- Li K, Mo P, Yang Y, Lu Y, Wen Y. *TADCluster: Temporal-Adaptive Density Clustering for Topic Identification in Online Question Answering Communities*.

Key external methods and benchmark references include:

- Reimers N, Gurevych I. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3980-3990. https://doi.org/10.18653/v1/D19-1410
- Grootendorst M. 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Jiang H, Beeferman D, Mao W, Roy D. 2024. Topic detection and tracking with time-aware document embeddings. *Proceedings of LREC-COLING 2024*, 16293-16303.
- Wu X, Nguyen T, Zhang DC, Wang WY, Luu AT. 2024. FASTopic: Pretrained Transformer is a fast, adaptive, stable, and transferable topic model. *Advances in Neural Information Processing Systems* 37:84447-84481.

## License and Contribution Guidelines

No separate software license has been assigned to this repository. The code is provided with the associated study for scholarly evaluation and research use. Questions about reuse can be directed to the corresponding author.

Contributions can be proposed through GitHub issues or pull requests. Keep changes focused, document any new dependencies, preserve the calibration/evaluation separation used by the pipeline, and do not commit raw datasets, generated result files, credentials, or local environment files.
