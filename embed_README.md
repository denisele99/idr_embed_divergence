# Generating Protein Language Model Embeddings

This script generates protein sequence embeddings from FASTA files using one of several protein language model (PLM) embedding methods.

## Purpose

Generate embeddings for protein or protein segment sequences (e.g. IDRs or domains) from FASTA files for downstream analyses such as neighbour divergence, clustering, UMAP visualization, or GO enrichment.

Supported embedding types:

* `esm`: embeddings from the ESM protein language model
* `IDR_LM`: embeddings from a trained IDR-specific language model checkpoint
* `IDR_LM_random`: embeddings from the same IDR language model architecture with randomly initialized weights

`IDR_LM_random` can be useful as a control to determine whether learned model weights contribute meaningful information beyond the model architecture itself.

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

## Embedding Types

### `esm`

Uses the ESM protein language model to generate embeddings from sequence.

Recommended when comparing against a general-purpose protein language model baseline.

### `IDR_LM`

Uses a trained IDR-specific language model checkpoint.

Requires:

```yaml
model_file: ../models/idr_lm_checkpoint.pt
```

Recommended when generating embeddings specialized for intrinsically disordered regions.

### `IDR_LM_random`

Uses the same IDR language model architecture, but with random weights instead of a trained checkpoint.

Useful as a negative control.

## Output

For each input FASTA file, the script generates an HDF5 (`.h5`) embedding file in `result_dir`.

Example output:

```text
results/embeddings/example_sequences.h5
```

The HDF5 file contains:

* `ids`: sequence identifiers from the FASTA file
* `embeddings`: embedding vectors corresponding to each sequence

Example structure:

```text
example_sequences.h5
├── ids
└── embeddings
```

## Example

Input FASTA:

```text
>HUMAN_Q14004_IDR_1_694
MSEQQR...

>HUMAN_P12345_IDR_50_120
MTPAAP...
```

Command:

```bash
python get_embeddings.py --config configs/embed_config.yaml
```

Output:

```text
results/embeddings/my_sequences.h5
```

containing:

* `ids = ["HUMAN_Q14004_IDR_1_694", "HUMAN_P12345_IDR_50_120"]`
* `embeddings = N x D embedding matrix`
