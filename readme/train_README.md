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
bert_config.json
config.py
IDRLM_pretrain_args.yaml
train.py
idrlm_embed.py
```

* `bert_config.json`
  Model architecture definition (hidden size, layers, attention heads, etc.).

* `config.py`
  Defines the training/config dataclasses and parses the YAML config.

* `idrlm_pretrain.yaml`
  Training configuration file containing model, dataset, and training parameters.

* `train.py`
  Main training script.

* `idrlm_embed.py`
  Generate embeddings from a trained IDR-LM checkpoint or from a randomly initialized model.

## Training

Train the model using:

```bash id="bykgio"
python idrlm_train.py --config idrlm_pretrain.yaml
```

## Training Config

Example `idrlm_pretrain.yaml`:

```yaml id="s3i0v4"
# logging
wandb_project: idr_lm
wandb_run_name: run1

# paths
#train/test split randomly from training file
train_data_path: ../data/seqs/human_idrs_pos_updated_labels.fasta #/home/moseslab/denise/IDR_LM/data/sequences/mobidb_idrs_27/mobi_idrs_30M.fasta #TODO change to sample dataset
model_config_path: ../src/idr_diverge/IDR_LM_train/bert_config.json
resume_checkpoint_path: null #Optional: load weights from a previous training run instead of starting from scratch. Set to None to initialize a new model from model_config_path
checkpoint_output_dir: ../res/model_checkpoints
final_model_dir: ../res/model_checkpoints/best
```

Paths are interpreted relative to the YAML config file.

## Outputs

Training writes model checkpoints and logs to `output_dir`.

Typical output:

```text id="xmd77t"
models/idr_lm/
├── checkpoint-1000/
├── checkpoint-2000/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── trainer_state.json
```

If `load_best_model_at_end=True`, the final saved model will be the best checkpoint according to the selected evaluation metric.

## Generating Embeddings

After training, generate embeddings using:

```bash id="y8u0g5"
python idrlm_embed.py \
    --seq-input path/to/sequences.fasta \
    --model-path models/idr_lm/checkpoint-2000 \
    --output-name example_embeddings
```

To generate embeddings from a randomly initialized model instead of a trained checkpoint, omit `--model-path` or use the random initialization mode if implemented in your script.
