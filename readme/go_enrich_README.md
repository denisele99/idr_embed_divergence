# GO Enrichment from Embedding Neighbours

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

## Config

```yaml id="ygojyc"
# GO ontology and annotation files
go_obo: ../data/annotations/go_2024.obo
go_annotations: ../data/annotations/uniprot_GO_annotations_2024.tsv

# Number of neighbours used for enrichment
k: 50

# Significance threshold after multiple-testing correction
alpha: 0.01

# Query IDs to analyze.
# Must exactly match IDs in target_embeddings.
query_ids: Q8IZL9_IDR_7_288, P21127_IDR_441_723

# If true, run enrichment for all IDs in target_embeddings
query_all: false

# Neighbour selection method:
# knn = nearest neighbours in embedding space
# BLAST = top BLAST hits
# RANDOM = random neighbours
search_method: knn

# Embedding file used to find neighbours.
# Typically human IDR or domain embeddings (.csv or .h5).
target_embeddings: ../data/embed/human_idrs_pos_updated_esm.csv

# BLAST output table, uses default format
# Required only when search_method = BLAST
target_blast_output: ../data/blast_res/human_idrs_pos_blast_out_names.tsv

# Output CSV containing enriched GO terms
output: ../res/go_enrich/knn_enrichment_goa_idrs.csv
```

## Required Inputs

* Embedding file containing the query regions and background dataset
* GO ontology (`.obo`)
* GO annotation table
* Optional BLAST output file if using `search_method: BLAST`

Query IDs must exactly match the identifiers in the embedding file.

## Output

The script writes a CSV table containing significantly enriched GO terms for each query region.

Example columns:

```text id="8m7rf8"
query_id,go_id,go_name,p_value,fdr,odds_ratio,n_neighbours
```

Example:

```text id="vrq3pw"
Q8IZL9_PFAM_7_288,GO:0005515,protein binding,1.2e-05,0.004,3.8,50
```

## Related Scripts

To evaluate or summarize enrichment results, see:

```text id="f0b7g5"
scripts/run_enrich_eval.py
```

##Notes:
/home/moseslab/denise/Paper/src/idr_diverge/go_enrichment/go_enrichment_.py and go_eval.py will break if configs/go_config.yaml is renamed or moved from configs (depends on go_obo and go_annotation file defined there)