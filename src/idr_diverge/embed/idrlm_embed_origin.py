
import sys
import os
import torch
import numpy as np
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample
import pandas as pd

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForMaskedLM,BertConfig,BertModel,BertTokenizer

'''Adapted from IDP-LM paper'''

sys.path.append('/home/moseslab/denise/IDR_LM/IDP_BERT')
#from pytorch_bert.modeling import BertModel
#from pytorch_bert.tokenization import BertTokenizer

import pretraining_args as args
from prepare_data import create_examples, convert_examples_to_features
from get_embedding import load_file_2_data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# File paths
seq_input_path = '/home/moseslab/denise/IDR_LM/data/mobidb_idrs/mobidb_idrs_1.fasta'  # Input: FASTA file
output_path = '/home/moseslab/denise/embeddings/IDR_LM/'  # Output: Directory for embeddings
model_path = '/home/moseslab/denise/IDR_LM/results/model_checkpoints/1M_8b_100ep-checkpoints/checkpoint-epoch-26/'  # Model checkpoint path


def load_file_2_data(file_path):
	loadfile = open(file_path,"r") 	
	load_f = []
	line_id = 1
	for line in loadfile:
		line=line.strip('\n')
		load_f.append(line)
		line_id += 1
	loadfile.close()

	load_data = []
	for i in range(len(load_f)):
		if i % 2 == 0:
			load_data.append(load_f[i:i+2])    #one data:  [0]--id  [1]--seq   
	# print("load_file: ",file_path,"    data length: ",len(load_data))  
	return load_data


def convert_sequences_to_inputs(sequences,tokenizer,vocab_list, max_seq_length):

	# print("Got your input sequences:",len(sequences))
	each_max = max_seq_length - 2
	
	new_data = []
	for i, sequence in enumerate(sequences):
		one_seq  = [r for r in sequence]
		s = 0
		for j in range(int(-(-len(one_seq)//each_max))):  #向上取整
			if s + each_max >= len(one_seq):
				end = len(one_seq) - s
				new_data.append(one_seq[s:s+end])
			elif s + each_max < len(one_seq):
				new_data.append(one_seq[s:s+each_max])
			s = s + each_max

	# 加标识符,没一个片都要加
	new_new_data = []
	for one_data in new_data:
		new_new_data.append(["[CLS]"] + one_data + ["[SEP]"])  # 加上[CLS],[SEP]标记


	# padding
	inputs = []
	for one_data in new_new_data:

		# 将输入、label转化为 idx
		input_ids = tokenizer.convert_tokens_to_ids(one_data)

		# 输入的 zero_padding 操作
		input_array = np.zeros(max_seq_length, dtype=int)
		input_array[:len(input_ids)] = input_ids

		# 实际输入长度的mask 向量
		mask_array = np.zeros(max_seq_length, dtype=int)
		mask_array[:len(input_ids)] = 1


		one_input = []
		one_input.append(input_array)
		one_input.append(mask_array)
		inputs.append(one_input)
	
	# print("covert finished, got totally:",len(inputs))

	return inputs



def get_encoding_from_model(sequences, inputs, model_path, config_path):
    """
    Generates embeddings from a pre-trained BERT model.

    Args:
        sequences (list): List of input sequences for which embeddings are to be generated.
        inputs (list): List of input tensors containing token IDs and attention masks for each sequence.
        model_path (str): Path to the pre-trained BERT model checkpoint.

    Returns:
        list: A list of embeddings for each input sequence, after removing padding and restoring original sequence lengths.
    """
    # Device configuration (use GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained BERT model
   
    #model = BertForMaskedLM.from_pretrained(model_path)
    
    if not(model_path):
        #Randomly initialized model
        config = BertConfig.from_json_file(config_path)
        model = BertForMaskedLM(config=config)
    
    else:
        model = BertModel.from_pretrained(model_path)
    
    model.to(device)
    model.eval()

    model.config.output_hidden_states = True #MODIFIED

    # Convert inputs to tensors
    all_input_ids = torch.tensor([f[0] for f in inputs], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in inputs], dtype=torch.long)
    
    # Create DataLoader
    test_data = TensorDataset(all_input_ids, all_input_mask)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    get_encodings = []

    # Encode sequences using BERT
    for input_ids, input_mask in test_dataloader:
        input_ids, input_mask = input_ids.to(device), input_mask.to(device)
        with torch.no_grad():
            #**********MODIFIED*************
            outputs = model(input_ids=input_ids, attention_mask=input_mask)
            last_layer_embeddings = outputs.hidden_states[-1]
            get_encodings.append(last_layer_embeddings.cpu().numpy())

            #encoded_outputs, pooled_output = model(input_ids=input_ids, attention_mask=input_mask)
            #get_encodings.append(encoded_outputs.cpu().numpy())
    # Method 2: Average the hidden states across all tokens to get a protein-level embedding
        
    # Combine batch encodings
    all_batch_encodings = np.concatenate(get_encodings, axis=0)

    org_lens_encoding = []
    each_max = args.max_seq_length - 2
    begin_idx = 0

    # Restore sequences from slices
    for seq in sequences:
        slice_num = int(np.ceil(len(seq) / each_max))
        if slice_num > 1:
            seq_encodings = all_batch_encodings[begin_idx:begin_idx + slice_num].reshape(-1, all_batch_encodings.shape[-1])
        else:
            seq_encodings = all_batch_encodings[begin_idx].reshape(-1, all_batch_encodings.shape[-1])
        begin_idx += slice_num
        org_lens_encoding.append(seq_encodings)

    final_encodings = []

    # Remove padding and restore original sequence lengths
    for i, seq in enumerate(sequences):
        slice_num = int(np.ceil(len(seq) / each_max))
        org_len = len(seq) + 2 * slice_num
        final_encodings.append(org_lens_encoding[i][:org_len])

    cleaned_encodings = []

    for i, seq in enumerate(sequences):
        slice_num = int(np.ceil(len(seq) / each_max))
        if slice_num <= 1:
            cleaned_encoding_1 = final_encodings[i][1:-1]
        else:
            del_idx = [j * args.max_seq_length for j in range(slice_num)] + \
                      [((j + 1) * args.max_seq_length) - 1 for j in range(slice_num - 1)] + \
                      [len(seq) + 2 * slice_num - 1]
            cleaned_encoding_1 = np.delete(final_encodings[i], del_idx, axis=0)
        
        mean_embedding = np.mean(cleaned_encoding_1, axis=0)
        cleaned_encodings.append(mean_embedding)

    return cleaned_encodings

    

def split_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def main(data, encoding_path, model_path,out_name='encodings'):
    """
    Processes a dataset and generates embeddings, saving them to files.

    Args:
        data (list): List of tuples where each tuple contains a sequence name and the corresponding sequence data.
        encoding_path (str): Directory where the generated embeddings will be saved.
        model_path (str): Path to the pre-trained BERT model checkpoint.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(encoding_path, exist_ok=True)
    batch_size =100

    csv_file_path = os.path.join(encoding_path, out_name+".csv")
    
    # Filter sequences that have not been encoded
    #for seq_name, seq_data in data:
    #    seq_name = seq_name.replace('>', '')
    #    if not os.path.exists(os.path.join(encoding_path, f"{seq_name}.npy")):
    #        inputs_sequences.append(seq_data)
    #        final_data.append((seq_name, seq_data))
    
    #if inputs_sequences:
    # Load vocabulary and tokenizer
    vocab_list = [line.strip("\n") for line in open(args.vocab_file, 'r')]
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

    #Split data into batches
    #print(len(data))
    data_batches = list(split_into_batches(data, batch_size))
    #print(len(data_batches))
    #print(data_batches)
    for i, data_batch in enumerate(data_batches):
        print(str(i),'/',str(len(data_batches)))
        #print(data_batch)
        
        inputs_sequences = []
        final_data = []
        
        for seq_name, seq_data in data_batch:
            seq_name = seq_name.replace('>', '')
            print(seq_name)
            inputs_sequences.append(seq_data)
            final_data.append((seq_name, seq_data))
            
        # Convert sequences to model inputs
        inputs = convert_sequences_to_inputs(inputs_sequences, tokenizer, vocab_list, args.max_seq_length)

        # Get model encodings
        encodings = get_encoding_from_model(inputs_sequences, inputs, model_path)

            # Save encodings to files
            #for seq_name, encoding in zip([fd[0] for fd in final_data], encodings):
            #    np.save(os.path.join(encoding_path, f"{seq_name}.npy"), encoding)
            
            # Prepare data for CSV
        csv_data = []
        for seq_name, encoding in zip([fd[0] for fd in final_data], encodings):
            flattened_encoding = encoding.flatten()  # Flatten the encoding array
            csv_data.append([seq_name] + flattened_encoding.tolist())
        #print(csv_data)
        # Convert to DataFrame
        df = pd.DataFrame(csv_data)

        # Save to CSV
        if os.path.exists(csv_file_path):
            df.to_csv(csv_file_path, mode='a',index=False, header=False)
        else:
            df.to_csv(csv_file_path, mode='w',index=False, header=False)

    print(f"Encodings saved to {csv_file_path}")

        

def get_embedding_IDP(data_file, file_path, model_path,output_name):
    """
    Loads data and generates embeddings for IDP sequences using a pre-trained BERT model.

    Args:
        data_file (str): Path to the data file containing the sequences.
        file_path (str): Directory where the generated embeddings will be saved.
        model_path (str): Path to the pre-trained BERT model checkpoint.

    Returns:
        None
    """
    # Load data and process sequences
    test_data = load_file_2_data(data_file)
    print("IDP-BERT processing sequences:", len(test_data))
    main(test_data, file_path, model_path,output_name)
    print("Done")




#use main with sys.argv
if __name__ == '__main__':

    out_dir = '/home/moseslab/denise/embeddings/IDR_LM'
    seq_input_path = sys.argv[1]
    if len(sys.argv)>2:
        model_path = sys.argv[2]
    else:
        model_path = None
    if len(sys.argv)>3:
        output_name = sys.argv[3]
    else:
        output_name = 'randominit_' + seq_input_path.split('/')[-1].split('.')[0]
    # Run the embedding generation process
    get_embedding_IDP(seq_input_path, output_path, model_path,output_name)
