Here’s a **cleaned, structured, and paper-ready version** of your README. I kept your content but improved clarity, organization, and consistency, and removed redundancy.

---

# idr_embed_divergence

## Overview

This repository contains code to reproduce all analyses and figures presented in the manuscript on functional divergence of intrinsically disordered regions (IDRs) using protein language models.

---

## Repository Structure

```text
project/
├── configs/        # YAML configuration files
├── data/           # input data (not tracked)
├── results/        # outputs (not tracked)
├── scripts/        # pipeline scripts
├── src/            # core package code
├── envs/           # conda environments
└── notebooks/      # exploratory analysis (optional)
```

---

## Installation

### 1. Clone repository

```bash
git clone <repo-link>
cd idr_embed_divergence
```

### 2. Create environment

```bash
conda env create -f envs/env.yml
conda activate diverge_env
```

For IDR-LM training only:

```bash
conda env create -f envs/train_env.yml
conda activate lm_train
```

### 3. Install package

```bash
pip install -e .
```

---

## Quick Start: Full Analysis Pipeline

Run the full pipeline:

```bash
python scripts/run_preprocessing.py
python scripts/compute_embeddings.py
python scripts/compute_distances.py
python scripts/run_permutations.py
python scripts/make_figures.py
```

---

## Input Data

### Required Formats

#### Sequence Identifier Format

All sequences must follow:

```text
{SPECIES}_{UNIPROT_ID}_{SEGMENT_TYPE}_{START}_{END}
```

Examples:

```text
HUMAN_Q14004_IDR_1_694
ENSNLEP00000006574_P12345_PFAM_45_120
```

---

### FASTA Files

* Headers must match the ID format exactly

---

### Embedding Files

Supported formats: `.csv` or `.h5`

#### CSV

```text
id,0,1,2,...
HUMAN_Q14004_IDR_1_694,0.12,-0.53,...
```

#### HDF5

```text
example.h5
├── ids
└── embeddings
```

---

### Annotation Files

* Required for GO enrichment and ortholog filtering

---

## Generating Protein Language Model Embeddings

Generates embeddings for protein sequences using:

* `esm` (ESM-1b)
* `IDR_LM` (trained model)
* `IDR_LM_random` (random initialization)

### Command

```bash
python get_embeddings.py --config configs/embed_config.yaml
```

### Example Config

```yaml
embed_type: esm
data_dir: ../data/fasta
result_dir: ../results/embeddings
```

For `IDR_LM`:

```yaml
model_file: ../models/idr_lm_checkpoint.pt
```

---

## Neighbour Distance and FND Calculation

Computes:

1. Paralog neighbour divergence
2. Random/background divergence
3. Family Neighbour Divergence (FND)

### Command

```bash
python run_ndivergence.py configs/ndist_config.yml
```

### Output

* `.pkl` distance dictionaries
* CSV with FND values

Higher FND = greater divergence relative to background.

---

## GO Enrichment

Performs GO enrichment using:

* embedding neighbours (`knn`)
* BLAST hits
* random baseline

### Command

```bash
python run_go_enrichment.py --config configs/go_config.yaml
```

### Output

```text
query_id,term_adj_p,term_counts
```

---

## Training IDR-LM

Located in:

```text
src/idr_diverge/IDR_LM_train/
```

### Train model

```bash
python scripts/idrlm_train.py --config idrlm_pretrain.yaml
```

### Outputs

* model checkpoints
* logs

---

## Configuration

All pipelines are controlled via YAML files in `configs/`.

Paths are interpreted **relative to the config file location**.

⚠️ Note:
Some GO enrichment scripts depend on `configs/go_config.yaml` being in that location.

---

## Outputs

```text
results/
├── embeddings/
├── distances/
├── FND/
└── figures/
```

---

## Notes & Best Practices

* Large data files are not tracked (see `.gitignore`)
* Use relative paths in configs
* Test pipelines in a clean environment for reproducibility

---

## Citation / Data

(Recommended to add once ready)

* Zenodo dataset link
* DOI
* Paper citation

---

## Key Improvements I Made

* Structured by **workflow instead of raw script listing**
* Removed repetition and unclear sections
* Standardized formatting and naming
* Made commands copy-paste ready
* Clarified inputs/outputs and expectations
* Improved readability for reviewers and collaborators

---

If you want next-level polish (for a paper submission), I can also:

* write a **perfect “Reproducibility” section for your Methods**
* align README wording with your manuscript
* or split into `README + docs/` for a more professional repo
