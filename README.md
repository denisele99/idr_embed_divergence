# idr_embed_divergence
# Paper Title

Code and data for:  
**Paper Title**  
Author et al., *Journal*, Year

## Overview
This repository contains code to reproduce all analyses and figures
presented in the manuscript.

## Contents
Data
1. Generating pLM embeddings
Neighbour Distance and Family Neighbour Divergence (FND) Calculation


## Requirements
- Python ≥ 3.10
- Conda / Mamba

environments included: envs/env.yml -> 
envs/train_env.yml -> environment for training IDR-LM only

```bash 
conda env create -f environment.yml
conda activate paper-env
```

## How to install
git clone [link]
inside repo: pip install -e . 

## FULL ANALYSIS PIPELINE FOR REPRODUCIBILITY

python scripts/run_preprocessing.py
python scripts/compute_embeddings.py
python scripts/compute_distances.py
python scripts/run_permutations.py
python scripts/make_figures.py

## INPUT DATA required format
- Fasta files for each protein sequence dataset (header format = >{species}_{Uniprot ID}_{segment type}_{region start}_{region end})
Assumes IDs look like 'SPECIES_..._...'; __gpos is the join of the last |pos| tokens.
- Embedding file: can be csv or h5 files
  (can generate from here - see generating plm embeddings)
  - if csv: first col is id name (same header format as fasta)
  - if h5: ids and embeddings as keys

- Annotation files for each protein sequence dataset

### Sequence Identifier Format
Most scripts assume a common identifier format across FASTA and embedding files.

All sequence IDs should follow:

```text id="l0e0li"
{SPECIES}_{UNIPROT_ID}_{SEGMENT_TYPE}_{START}_{END}
```

 SPECIES is "HUMAN" OR ENSEMBL identifier

Example:

```text id="bslzgh"
HUMAN_Q14004_IDR_1_694
ENSNLEP00000006574_P12345_PFAM_45_120
```

### FASTA Format

FASTA headers must exactly match the ID format above:


### Embedding File Format

Embedding files may be `.csv` or `.h5`.

(These can be generated using the embedding generation script.)

#### CSV

* first column = sequence ID
* remaining columns = embedding values

```text id="6ayebk"
id,0,1,2,...
HUMAN_Q14004_IDR_1_694,0.12,-0.53,...
```

#### HDF5 (`.h5`)

Example structure:

```text id="hylrjx"
example.h5
├── ids
└── embeddings
```

# Generating Protein Language Model Embeddings

This script generates protein sequence embeddings from FASTA files using select protein language model (pLM) embedding methods.

## Purpose

Generate embeddings for protein or protein segment sequences (e.g. IDRs or domains) from FASTA files for downstream analyses such as neighbour divergence, clustering, UMAP visualization, or GO enrichment.

Supported embedding types:

* `esm`: embeddings from the ESM-1b protein language model
* `IDR_LM`: embeddings from a trained IDR-specific language model checkpoint
* `IDR_LM_random`: embeddings from the same IDR language model architecture with randomly initialized weights


## Input

The script accepts either:
* a single FASTA file
* a directory containing multiple FASTA files

specified through `data_dir`.

Each FASTA entry should contain a unique sequence identifier and sequence, for example:

```text
>HUMAN_Q14004_IDR_1_694
MSEQQR...
```

### Required Inputs

* FASTA file or directory of FASTA files
* Embedding type (`esm`, `IDR_LM`, or `IDR_LM_random`)

### Additional Required Input for `IDR_LM`

If using `IDR_LM`, you must also provide:

* `model_file`: path to the trained model checkpoint

## Configuration

Example `configs/embed_config.yaml`:

```yaml
embed_type: esm

esm_script: ../src/idr_diverge/embed/esm_embed.py
idr_lm_script: ../src/idr_diverge/embed/idrlm_embed.py

data_dir: ../data/fasta
result_dir: ../results/embeddings

# Only required for embed_type: IDR_LM
model_file: ../models/idr_lm_checkpoint.pt
```
Paths in the config file are interpreted relative to the config file location.

## Command

Run using a config file:

```bash
python get_embeddings.py --config configs/embed_config.yaml
```

You can also override config values from the command line:

```bash
python get_embeddings.py \
    --config configs/embed_config.yaml \
    --embed-type IDR_LM \
    --model-file models/best_model.pt
```

## Output

For each input FASTA file, the script generates an HDF5 (`.h5`) embedding file in `result_dir`.

The HDF5 file contains:

* `ids`: sequence identifiers from the FASTA file
* `embeddings`: embedding vectors corresponding to each sequence


# Neighbour Distance and Family Neighbour Divergence (FND) Calculation

Pipeline calculates neighbour distance divergence between paralogous protein regions and compares it to an ortholog-based background to compute Family Neighbour Divergence (FND).

### Overview

1. Compute neighbour distances between paralogous segments within each family (`neighbour_divergence`)
2. Optionally compute neighbour distances for randomly sampled segments (`random_neighbour_divergence`)
3. Compare family distances to the background distances to calculate FND (`calculate_FND`) -> must first run 1. and 2., then set outputs as arguments and set calculate_FND to true

FND is defined as the difference between the average paralog neighbour distance and the expected ortholog/background neighbour distance.

---

## Command

```bash id="0kjxzk"
python run_ndivergence.py ../configs/ndist_config.yml
```

---

## Input files

genes of families to embed:
family_to_genes.json
{"CDKs": {GENE1, GENE2, ...}}


## Config

```yaml id="a92d9x"
inputs:
  # Embedding file used as the background space for neighbour-rank calculations.
  # May be .csv or .h5. Typically all human IDR or domain embeddings.
  background_emb: ../data/embed/human_idrs_pos_updated_esm_h_prefix.csv

  # Ortholog embedding file or directory of ortholog embedding files (.h5 or .csv).
  # Used to compute ortholog neighbour distances and/or family divergence.
  ortholog_emb: ../data/embed/orthologs/

  # JSON mapping from family name -> list of genes.
  # Required unless only running random_neighbour_divergence.
  fam_map: ../src/idr_diverge/distances/input_args/family_to_genes.json

outputs:
  # Output file or directory for family neighbour-distance results.
  out_distance: ../res/ndist/fam_ndist

compute:
  # Number of pairwise comparisons processed at once.
  # Increase for speed, decrease if running out of memory.
  chunk_size: 8000

ortholog_filter:
  enabled: true

  # Which Ensembl ortholog types to keep.
  # Options: ortholog_one2one, ortholog_one2many, ortholog_one2one_one2many
  ortholog_types: ortholog_one2one_one2many

  # Minimum fraction of paralog segments that must contain a species.
  min_species_coverage: 0.3

  # Minimum number of species required after filtering.
  min_species: 10

  # Ensembl ortholog annotation file used for filtering.
  homolog_annotation: ../data/annotations/ensembl_homol_reference_10-10.pkl

neighbour_divergence:
  # Distance type:
  # between = pairwise distances between different paralogous segments
  # within = pairwise distances among orthologs of the same segment
  dist_type: between

  # Segment label used only for output naming.
  segment: IDR

  # For within divergence only:
  # per_gene = return distance per gene for each species
  # aggregate = return one distance value per species
  
  return_mode: per_gene

random_neighbour_divergence:
  enabled: true

  # Number of random segments sampled for each background estimate.
  sample_size: 100

  # Optional random seed for reproducibility.
  random_seed: 42

  # Output file for random/background neighbour-distance results.
  out_random: ../res/ndist/rand_ndist/random_idr_diverge

calculate_FND:
  enabled: true

  # Background neighbour-distance file (.pkl) or directory.
  # If omitted, background distances will be recomputed from random samples.
  bg_distance_matrix: ../res/ndist/rand_ndist/random_idr_diverge.pkl

  # Family neighbour-distance file or directory produced by neighbour_divergence.
  fam_distance_matrix: ../res/ndist/fam_ndist

  # Output CSV containing FND values.
  output_results: ../res/FND/fnd_results.csv
```

---



## Output Files

### Neighbour Distance Output

`out_distance` and `out_random` produce `.pkl` files containing neighbour distance dictionaries.

For `dist_type: between`, values are typically stored per family and species:

```text id="d5i9kl"
family -> species -> neighbour distance
```

For `dist_type: within`, values are stored per segment/gene.

EXAMPLE OF OUTPUT DICT

{GENE1_IDR_1 | GENE2_IDR_1: {SPP: Distance, SPP2: Distance} }
{'O94921_IDR_1_130|P21127_IDR_1_420': {'ENSABRP': [5712.5, 5714.5], 'ENSACCP': [5607.0, 5716.0], .. }}


### FND Output

`output_results` is a CSV containing one row per family (or family/species pair), for example:

```text id="5q2n3s"
Group_id,log_mean Divergence_per_ortholog,FND,1samp_ttest_stat,1samp_ttest_pvalue #TODO change example
100,{'ENSCSEP': 2.10797268245514, 'ENSPFOP': 1.7439702704850224},2.15,49.75,1.15e-108
```

Higher positive FND values indicate paralogous segments are more diverged than expected from the ortholog/background baseline.



# GO Enrichment from Embedding Neighbours (or BLAST hits)

This script identifies significantly enriched GO terms for protein regions by performing GO enrichment on their nearest neighbours in embedding space.

Neighbours can be selected using:

* `knn`: k nearest neighbours in pLM embedding space (default)
* `BLAST`: top k BLAST hits
* `RANDOM`: random k neighbours as a background/control

Default behaviour uses 50 nearest neighbours (`k = 50`).

## Command

```bash id="ccjlwm"
python run_go_enrichment.py --config configs/go_config.yaml
```

## Required Inputs

Specified inside config:

* Embedding file containing the query regions and background dataset
* GO ontology (`.obo`)
* GO annotation table
* Optional BLAST output file if using `search_method: BLAST`

Query IDs must exactly match the identifiers in the embedding file.

## Output

The script writes a CSV table containing significantly enriched GO terms for each query region.
includes dictionaries of enriched GO terms and their adjusted p-values, and dictionary with their term counts in k neighbours / background term count
Example columns:

```text id="8m7rf8"
query_id,term_adj_p,term_counts
```

Example:

```text id="vrq3pw"
Q96ND8_IDR_60_211,{'GO:0006357': 2.23e-14, 'GO:0019219': 1.38e-11, ...}, {'GO:0006355': '20/7717', 'GO:2001141': '20/7774'}
```

## Related Scripts

To evaluate or summarize enrichment results, see:

```text id="f0b7g5"
scripts/run_enrich_eval.py
```

##Notes:
src/idr_diverge/go_enrichment/go_enrichment_.py and go_eval.py will break if configs/go_config.yaml is renamed or moved from configs (depends on go_obo and go_annotation file defined there)



# Training IDR-LM

This directory contains code for training the IDR-LM protein language model and generating embeddings from the trained model.

```text id="7p64nl"
src/idr_diverge/IDR_LM_train/
```

## Requirements

- install lm_train environment (see train_env yaml)
- conda activate lm_train

## Files

```text id="8ikn9h"
- bert_config.json -> model config
- config.py -> structure of config 
- idrlm_pretrain.yaml -> pretrain args
- idrlm_train.py -> main training script
- idrlm_embed.py -> generate IDR_LM embeddings from pretrained/model checkpoint or randomly initialized model
```

## Training command

Train the model using:

```bash id="bykgio"
python scripts/idrlm_train.py --config idrlm_pretrain.yaml
```

NOTE: Paths defined in config are interpreted relative to the YAML config file.


## Outputs

Training writes model checkpoints and logs to `output_dir`.

If `load_best_model_at_end=True`, the final saved model will be the best checkpoint according to the selected evaluation metric.

## Relevant notebooks

TBD

## Data

Raw data are available from [Zenodo link].
Example processed datasets are provided in data/processed/.


---

