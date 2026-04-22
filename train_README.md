# Training IDR-LM

This directory contains code for training the IDR-LM protein language model and generating embeddings from the trained model.

```text id="7p64nl"
src/idr_diverge/IDR_LM_train/
```

## Requirements

- install lm_train environment (see env yaml)
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

* `IDRLM_pretrain_args.yaml`
  Training configuration file containing model, dataset, and training parameters.

* `train.py`
  Main training script.

* `idrlm_embed.py`
  Generate embeddings from a trained IDR-LM checkpoint or from a randomly initialized model.

## Training

Train the model using:

```bash id="bykgio"
python train.py --config IDRLM_pretrain_args.yaml
```

## Training Config

Example `IDRLM_pretrain_args.yaml`:

```yaml id="s3i0v4"
train_file: ../data/train.txt
validation_file: ../data/valid.txt

output_dir: ../models/idr_lm

num_train_epochs: 35
per_device_train_batch_size: 16
learning_rate: 5e-5

bert_config: ../src/idr_diverge/IDR_LM_train/bert_config.json
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
