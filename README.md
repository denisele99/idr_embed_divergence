# idr_embed_divergence
# Paper Title

Code and data for:  
**Paper Title**  
Author et al., *Journal*, Year

## Overview
This repository contains code to reproduce all analyses and figures
presented in the manuscript.

## Contents
1.

## Requirements
- Python ≥ 3.10
- Conda / Mamba

```bash 
conda env create -f environment.yml
conda activate paper-env
```

## How to install
git clone [link]
inside repo: pip install -e . (name of thing)

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


## Required Input Data Format

Most scripts assume a common identifier format across FASTA and embedding files.

### Sequence Identifier Format

All sequence IDs should follow:

```text id="l0e0li"
{SPECIES}_{UNIPROT_ID}_{SEGMENT_TYPE}_{START}_{END}
```

Example:

```text id="bslzgh"
HUMAN_Q14004_IDR_1_694
MOUSE_P12345_DOMAIN_45_120
```

Scripts assume:

* species = first token (`SPECIES`)
* region/position (`__gpos`) = last tokens, e.g.

```python id="pn4t62"
"_".join(id.split("_")[-4:])
# Q14004_IDR_1_694
```

### FASTA Format

FASTA headers must exactly match the ID format above:

```text id="m4djx0"
>HUMAN_Q14004_IDR_1_694
MSEQQRQE...

>MOUSE_P12345_PFAM_45_120
MTAPGAA...
```

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

Must contain datasets:

```text id="fdv51g"
ids
embeddings
```

Example structure:

```text id="hylrjx"
example.h5
├── ids
└── embeddings
```

where:

* `ids` = list of sequence IDs
* `embeddings` = matrix of embedding vectors

The number of `ids` must equal the number of embedding rows.


## Generating plm embeddings



# Neighbour Distance and Family Neighbour Divergence (FND) Calculation

This pipeline calculates neighbour distance divergence between paralogous protein regions and compares it to an ortholog-based background to compute Family Neighbour Divergence (FND).

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


# GO enrichment for IDR function annotation - 50-NN GO enrichment



## Training IDR-LM
-src/idr_diverge/IDR_LM_train
codes:
- bert_config.json -> model config
- config.py -> structure of config 
- IDRLMpretrain_args.yaml -> pretrain args
- train.py _>
- idrlm_embed.py -> get IDR_LM embeddings from pretrained or randomly initialized model

Command:

python train.py --config IDRLM_pretrain_args.yaml



## Visualization of divergence relative to other protein families?

## Relevant notebooks

## Data

Raw data are available from [Ensembl / UniProt / Zenodo link].
Processed datasets are provided in data/processed/.


---

## 3. CITATION.cff (strongly recommended)

This enables **one-click citation on GitHub**.

```yaml
cff-version: 1.2.0
title: "Paper Title"
authors:
  - family-names: Le
    given-names: Denise
  - family-names: ...
date-released: 2025-06-01
version: 1.0.0
doi: 10.XXXX/zenodo.XXXX
