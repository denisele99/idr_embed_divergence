# idr_embed_divergence
# Paper Title

Code and data for:  
**Paper Title**  
Author et al., *Journal*, Year

## Overview
This repository contains code to reproduce all analyses and figures
presented in the manuscript.

## Contents

## Requirements
- Python ≥ 3.10
- Conda / Mamba

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
- Annotation files for each protein sequence dataset

insert EXAMPLE of data file


## GO enrichment for IDR function annotation

For a given config-file, GO term prediction can be performed with the following command:

python predict_go_embedding_inference.py config.txt

python run_go_enrichment.py --config /home/moseslab/denise/Paper/configs/go_config.yaml

Other scripts:
- run eval from res /home/moseslab/denise/Paper/scripts/run_enrich_eval.py
- 

Option to run BLAST search?
Blast default results as input




## Calculating Neighbour distance, Calculating FND
Required format for fasta

## Visualization of divergence relative to other protein families?

## Figures
Figure 1: results/figures/figure1.pdf
Figure 2: results/figures/figure2.pdf


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
