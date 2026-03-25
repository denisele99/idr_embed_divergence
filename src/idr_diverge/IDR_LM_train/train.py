# @Last Modified time: 2024-07-04
from __future__ import absolute_import
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append('/home/moseslab/denise/IDR_LM/IDP_BERT')
sys.path.append('/home/moseslab/denise/IDR_LM/src/general')


import numpy as np
import torch


from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM,BertConfig,BertModel,BertTokenizer

#from pytorch_bert.tokenization import BertTokenizer
from pytorch_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME
#from pytorch_bert.modeling import BertForMaskedLM, BertConfig, BertModel
from get_embedding import convert_sequences_to_inputs
from helper_functions import load_fasta_as_dataset,tokenize_dataset

import pretraining_args as args
from datasets import load_dataset
from transformers import TrainerCallback

#CUDA_LAUNCH_BLOCKING=1

import time

start = time.time()
print(start)



#load config file

#logger.info("Model config {}".format(config))


#1. split datasets for training
data_file_path = '/home/moseslab/denise/IDR_LM/data/sequences/mobidb_idrs/mobi_idrs_2_0.fasta'#'/home/moseslab/denise/IDR_LM/data/mobidb_ids_3M_filter_2.fasta'#'/home/moseslab/denise/IDR_LM/data/IDP_BERT_pretrain_dataset.txt' #'/home/moseslab/denise/IDR_LM/data/mobidb_ids_3M_filter_2.fasta'
dataset = load_fasta_as_dataset(data_file_path)

#TEMPORARY TODO
dataset = dataset.shuffle(seed=42)#.select(range(1000000))#range(100000))

# split the dataset into training (90%) and testing (10%)
dataset = dataset.train_test_split(test_size=0.15)
#2. Initialize tokenizer
#tokenizer = BertTokenizer(vocab_file=args.vocab_file)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert") #This tokenizer has an extra letter (O)

##Tokenize dataset (train and test)
print('Tokenize')

def tokenize_function(examples):
    examples = [' '.join(seq) for seq in examples['sequences']]
    return tokenizer(examples, padding='max_length', truncation=True, max_length=args.max_seq_length) 
#tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["id", "sequences"])
tokenized_dataset = tokenize_dataset(dataset,tokenizer,max_length=512)


train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]
#train_dataset = tokenized_dataset["train"].shuffle(seed=42)#.select(range(1000))  # Use a subset for quick training
#eval_dataset = tokenized_dataset["test"].shuffle(seed=42)#.select(range(1000)) 

#print(train_dataset[:10])

#print(time.time()-start)


#Define datacollator (mask tokens, default mask 15% of tokens)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.05 #probability in pretraining args is 0.05
)

#Load model
print('Get model')
# Initialize the model from a configuration without pretrained weights
config_file = '/home/moseslab/denise/IDR_LM/IDP_BERT/bert_config_2.json'
config = BertConfig.from_json_file(config_file)

model = BertForMaskedLM(config=config)


checkpoint_path = '/home/moseslab/denise/IDR_LM/results/model_checkpoints/1M_8b_100ep-checkpoints/checkpoint-epoch-26'
model_checkpoint = BertForMaskedLM.from_pretrained(checkpoint_path,config=config)


#Define training arguments

#from https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
batch_size = 64 #2048 #1024 #512 #256 #64
# Show the training loss with every epoch
logging_steps = len(dataset["train"]['sequences']) // batch_size
#model_name = model_checkpoint.split("/")[-1]



class SaveCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        #args.output_dir
        output_dir = os.path.join('/home/moseslab/denise/IDR_LM/results/model_checkpoints', f"checkpoint-epoch-{int(epoch)}")
        kwargs['model'].save_pretrained(output_dir)
        #kwargs['tokenizer'].save_pretrained(output_dir)
        print(f"Saved model checkpoint to {output_dir}")

print('Set training arguments')

training_args = TrainingArguments(
    #output_dir=args.output_dir,
    output_dir='../results',
    overwrite_output_dir=True,
    do_train = True,
    evaluation_strategy = 'epoch',
    num_train_epochs= 3,#args.num_train_epochs,
    learning_rate=args.learning_rate_1,
    no_cuda=args.no_cuda,
    per_device_train_batch_size=8,#args.train_batch_size,
    per_device_eval_batch_size=8,#args.eval_batch_size,
    save_steps=8192, #not mentioned
    eval_steps=4096,
    save_total_limit=10,
    logging_dir='./logs',
    logging_steps=logging_steps,
    warmup_ratio=args.warmup_proportion,

    #load_best_model_at_end=True,
    save_strategy='epoch'
    
)

#Create trainer for model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[SaveCheckpointCallback]
    #prediction_loss_only=True,
)

# Check tokenized dataset sample
#print(tokenized_dataset['train'][0])

# Add assertions to check data shapes and types
assert len(tokenized_dataset['train'][0]['input_ids']) == 512
assert all(len(input) <= 512 for input in train_dataset['input_ids']), "Input sequence exceeds 512 tokens"
#assert tokenized_dataset['train'][0]['input_ids'].shape == (512,)
#assert type(tokenized_dataset['train'][0]['input_ids']) == torch.int64

#Train the model
trainer.train()



#Check trained model using pipeline
import math
eval_results = trainer.evaluate()
print(eval_results)

#Save model
trainer.save_model("./saved_model/best_model")

end = time.time()
print(end - start)
