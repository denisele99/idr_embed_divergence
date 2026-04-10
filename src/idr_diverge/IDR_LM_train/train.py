from __future__ import absolute_import
import wandb
import os
from pathlib import Path
import random
import numpy as np
import time
from datasets import Dataset, DatasetDict, load_dataset

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from idr_diverge.utils.helpers import read_fasta

# ---------------------------
# Configuration
# ---------------------------

WANDB_PROJECT = "idr_lm_test1"
WANDB_RUN_NAME = "test1"



# FASTA file containing the protein sequences used for pretraining
TRAIN_DATA_PATH = Path("/home/moseslab/denise/IDR_LM/data/sequences/mobidb_idrs_27/mobi_idrs_30M.fasta")
# BERT architecture configuration (hidden size, layers, attention heads, etc.)
MODEL_CONFIG_PATH = Path("./bert_config.json") 

# Optional: load weights from a previous training run instead of starting from scratch.
# Set to None to initialize a new model from MODEL_CONFIG_PATH.
RESUME_CHECKPOINT_PATH = None #Path("/home/moseslab/denise/IDR_LM/results/model_checkpoints/30M_64b_5ep_-checkpoints-08-28/checkpoint-228180")

# Directory where training checkpoints and intermediate model saves will be written
CHECKPOINT_OUTPUT_DIR = Path("/home/moseslab/denise/Paper/res/model_checkpoints")

#OUTPUT_DIR = Path("/home/moseslab/denise/IDR_LM/results")
FINAL_MODEL_DIR = Path("/home/moseslab/denise/Paper/res/model_checkpoints/best")

#Training arguments
MAX_LENGTH = 512
BATCH_SIZE = 64
TEST_SIZE = 0.15
SHUFFLE_SEED = 42
MLM_PROBABILITY = 0.05
NUM_TRAIN_EPOCHS = 2 #35
LEARNING_RATE = 1e-4
NO_CUDA = False
WARMUP_PROPORTION = 0.1



# ---------------------------
# Helpers
# ---------------------------

def combine_sequences(sequences, tokenizer, max_length=512):
    """
    Combines multiple sequences into a list of tokenized sequences with a specified maximum length.
    
    Args:
        sequences (list of str): List of input sequences to be tokenized and combined.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the sequences.
        max_length (int): The maximum length of the combined token sequences.
    
    Returns:
        list of list of int: List of tokenized sequences, each within the specified maximum length.
    """
    combined_sequences = []  # List to store combined tokenized sequences
    combined_tokens = [tokenizer.cls_token_id]  # Start with the CLS token
    combined_attention = []
    current_length = 1  # Account for the CLS token

    for seq in sequences:
        seq = ' '.join(seq)
        tokens = tokenizer.encode(seq, add_special_tokens=False, truncation=True, max_length=max_length-2)  # Tokenize without special tokens
        tokens.append(tokenizer.sep_token_id)  # Append SEP token

        if current_length + len(tokens) > max_length:
            # If adding the current tokens exceeds max_length, pad and save the current sequence
            remaining_space = max_length - current_length
            combined_tokens.extend([tokenizer.pad_token_id] * remaining_space)  # Pad the remaining space
            #combined_sequences.append(combined_tokens)
            
            attention_mask = [1] * current_length + [0] * remaining_space
            combined_attention.append(attention_mask)
            combined_sequences.append(combined_tokens)

            # Reset for the next sequence
            combined_tokens = [tokenizer.cls_token_id] + tokens
            current_length = len(combined_tokens)
        else:
            combined_tokens.extend(tokens)
            current_length += len(tokens)

    # Append the last combined sequence
    remaining_space = max_length - current_length
    combined_tokens.extend([tokenizer.pad_token_id] * remaining_space)  # Pad the remaining space

    attention_mask = [1] * current_length + [0] * remaining_space

    combined_sequences.append(combined_tokens)
    combined_attention.append(attention_mask)
    combined_dict = {
            'input_ids': combined_sequences,
            'attention_mask': combined_attention
            
    }
    return combined_dict

def tokenize_dataset(dataset:DatasetDict,tokenizer:BertTokenizer,max_length=512) -> DatasetDict:
    #Appends sequences together to fill up the context length
    #Assumes test/train split
    
    train_input = [' '.join(seq) for seq in dataset['train']['sequences']]
    test_input = [' '.join(seq) for seq in dataset['test']['sequences']]
    
    train_tokens  = combine_sequences(train_input, tokenizer, max_length)
    test_tokens  = combine_sequences(test_input, tokenizer, max_length)
    
    train_tokenized_dataset = Dataset.from_dict(train_tokens)
    test_tokenized_dataset = Dataset.from_dict(test_tokens)
    
    tokenized_dataset = DatasetDict({
    'train': train_tokenized_dataset,
    'test': test_tokenized_dataset
    })
    
    return tokenized_dataset

def load_fasta_as_dataset(file_path):
    sequences = read_fasta(file_path)
    return Dataset.from_list(sequences)

class SaveCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        output_dir = CHECKPOINT_OUTPUT_DIR  / f"checkpoint-epoch-{epoch}"
        kwargs["model"].save_pretrained(output_dir)
        print(f"Saved model checkpoint to {output_dir}")



def initialize_wandb():
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

def load_and_tokenize_dataset(data_file: Path, tokenizer: BertTokenizer) -> DatasetDict:
    dataset = load_fasta_as_dataset(str(data_file))
    dataset = dataset.shuffle(seed=SHUFFLE_SEED).select(range(1000))
    dataset = dataset.train_test_split(test_size=TEST_SIZE)

    print("Tokenizing dataset...") 
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=MAX_LENGTH)
    
    return tokenized_dataset

def load_model(config_file: Path, checkpoint_path: Path | None = None) -> BertForMaskedLM:
    config = BertConfig.from_json_file(str(config_file))

    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = BertForMaskedLM.from_pretrained(str(checkpoint_path), config=config)
    else:
        print("Initializing model from config") # Initialize the model from a config without pretrained weights
        model = BertForMaskedLM(config=config)

    return model

def build_training_args(logging_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(CHECKPOINT_OUTPUT_DIR),
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        no_cuda=NO_CUDA,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=10,
        logging_dir=CHECKPOINT_OUTPUT_DIR / "logs",
        logging_steps=logging_steps,
        warmup_ratio=WARMUP_PROPORTION,
        
        load_best_model_at_end=True, #TODO hide?
        report_to="wandb", #TODO - log to wandb as an option?
    )


def validate_tokenized_dataset(tokenized_dataset):
    assert len(tokenized_dataset["train"][0]["input_ids"]) == MAX_LENGTH
    assert all(
        len(input_ids) <= MAX_LENGTH for input_ids in tokenized_dataset["train"]["input_ids"]
    ), "Input sequence exceeds max length"


def main():
    start = time.time()
    print(f"Start time: {start}")

    initialize_wandb()
    #tokenizer = BertTokenizer(vocab_file=args.vocab_file)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert") #This tokenizer has an extra letter (O)
    tokenized_dataset = load_and_tokenize_dataset(TRAIN_DATA_PATH , tokenizer)

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    validate_tokenized_dataset(tokenized_dataset)

    print(f"Dataset prep time: {time.time() - start:.2f} seconds")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    model = load_model(MODEL_CONFIG_PATH, RESUME_CHECKPOINT_PATH)

    logging_steps = max(1, len(train_dataset) // BATCH_SIZE)
    training_args = build_training_args(logging_steps=logging_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[SaveCheckpointCallback()] #TODO remove this? was originally hidden
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation results:")
    print(eval_results)

    trainer.save_model(str(FINAL_MODEL_DIR))

    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
