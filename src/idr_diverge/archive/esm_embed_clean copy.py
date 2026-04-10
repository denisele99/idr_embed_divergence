from csv import writer
import pandas as pd
import multiprocessing
import glob
import torch
import numpy as np
from Bio import SeqIO
import sys
import h5py
import numpy as np


''''
Code adapted from Bioembeddings, modified by me
https://github.com/facebookresearch/esm/blob/dfa524df54f91ef45b3919a00aaa9c33f3356085/README.md#quick-start-'''




def read_fasta(file:str)-> list: #return list of SeqRecord Objects without the alignments
    all_rec=[]
    for rec in SeqIO.parse(file,"fasta"):
        rec.seq=rec.seq.replace('-','')
        all_rec.append(rec)
    return all_rec




out_path = '/home/moseslab/denise/embeddings/esm1b/'

model, alphabet = torch.hub.load("facebookresearch/esm:main","esm1b_t33_650M_UR50S")
batch_converter = alphabet.get_batch_converter()
model = model.eval()


def get_esm_model():
    
    global model, alphabet, batch_converter

    if torch.cuda.is_available():
        model.to('cuda')
        #model.half() ?
    
    else:
        torch._C._jit_set_profiling_mode(False)
        torch.set_num_threads(multiprocessing.cpu_count())
        model = torch.jit.freeze(model)
        model = torch.jit.optimize_for_inference(model)

    
    return model, batch_converter

def esm_embed(seq): #per-residue embeddings for a single sequence

    model, batch_converter = get_esm_model()

    _, _, toks = batch_converter([("prot",seq)])
    #batch_labels, batch_strs, batch_tokens = batch_converter(data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #toks = toks.to(device=device, non_blocking=True)
    
    #model.half() if device=='cuda' else model.full()

    if torch.cuda.is_available():
        device = 'cuda'
        toks = toks.to(device=device, non_blocking=True)
        
        #else keep model and tensors on the cpu
    
    with torch.no_grad():
        results = model(toks, repr_layers=[33]) #layer 33 specific for esm

    token_representations = results["representations"][33].to(device="cpu")[0].detach().numpy() #is the to.device necessary?
    
    token_representations = token_representations[1:-1] #gets rid of first and last tokens (cls and eos)

    return token_representations

def get_esm_embeddings(fasta_path,out_name,dir = False, per_protein=False):

    if dir: #If path is a directory
        sequences = []
        fasta_files = glob.glob(fasta_path+'/*')
        for fasta_file in fasta_files:
            sequences.extend(read_fasta(fasta_file))

    else:
        sequences = read_fasta(fasta_path)

    #Prepare dataset, enumerate something, create batches, use batch_encoder??
    if '/' in out_name:
        full_out_path=out_name+'.csv'
    else:
        full_out_path = out_path+out_name+'.csv'

    pd.DataFrame().to_csv(full_out_path, mode = 'w', header=False)

    

    for sequence in sequences:
        seq = str(sequence.seq)
        l = len(seq)
        embtoks = None

        if l > 1022:
            piece = int(l/1022)+1
            part = l/piece
            for i in range(piece):
                st = int(i*part)
                sp = int((i+1)*part)
                results = esm_embed(seq[st:sp])
                
                if embtoks is not None:
                    embtoks = np.concatenate((embtoks,results), axis = 0)
                    #for i in range(len(results)):
                    #    embtoks[i] = np.concatenate((embtoks[i][:len(embtoks[i])-1],results[i][1:]),axis=0)

                else:
                    embtoks = results
            #merge embtoks?

        else:
            embtoks = esm_embed(seq)

        reduced_embedding = embtoks.mean(0)
        #for embedding in embtoks:
        #reduced_embedding.append(embedding.mean(0))

        embed_df = pd.DataFrame(reduced_embedding.reshape(1,-1),index=[sequence.name])
        #append to csv file
        embed_df.to_csv(full_out_path, mode= 'a', index=True, header=False)
        
    return full_out_path



    
    


dim=1280 #ESM-1b embedding dimension
batch_size = 100 #Number of sequences to process in each batch

def get_esm_embeddings_to_hd5(fasta_path,out_name,dir = False, per_protein=False):
    """
    Function to get ESM embeddings from a FASTA file or directory of FASTA files and save them to an HDF5 file.
    Parameters:
    fasta_path (str): Path to the FASTA file or directory containing FASTA files.
    out_name (str): Name of the output HDF5 file (without extension).
    dir (bool): If True, treat fasta_path as a directory containing multiple FASTA files.
    per_protein (bool): If True, generate per-sequence representations via averaging.
    Returns:
    str: Path to the output HDF5 file containing embeddings.
    """
    if dir: #If path is a directory
        sequences = []
        fasta_files = glob.glob(fasta_path+'/*')
        for fasta_file in fasta_files:
            sequences.extend(read_fasta(fasta_file))

    else:
        sequences = read_fasta(fasta_path)
    
    #Prepare dataset, enumerate something, create batches, use batch_encoder??
    if '/' in out_name:
        full_out_path=f"{out_name}.h5"
    else:
        full_out_path = f"{out_path}{out_name}.h5"


    # Create an HDF5 file to store embeddings
    with h5py.File(full_out_path, "w") as f:
        f.create_dataset("embeddings", shape=(0, dim), maxshape=(None, dim),
                        dtype="float32", chunks=True, compression="gzip")
        f.create_dataset("ids", shape=(0,), maxshape=(None,), dtype="S100",
                        chunks=True, compression="gzip")
    
    #split into batches?
    def get_batches(data,batch_size):
        for i in range(0, len(data), batch_size):
            #size = min(i+batch_size, len(data) - i)
            data_batch = data[i:i + batch_size]
            yield i,data_batch
    
    n_batch = len(sequences)/ batch_size #number of batches
    n=1
    
    for start_idx,sequences in get_batches(data=sequences,batch_size=batch_size):
        print(f"Processing batch {n}/{n_batch} with {len(sequences)} sequences...")
        n+=1
        # Process each batch of sequences
        # Initialize an empty array to hold the reduced embeddings
        batch_embeddings =np.empty((0, dim))
        for sequence in sequences:
            seq = str(sequence.seq)
            l = len(seq)
            embtoks = None

            if l > 1022:
                piece = int(l/1022)+1
                part = l/piece
                for i in range(piece):
                    st = int(i*part)
                    sp = int((i+1)*part)
                    results = esm_embed(seq[st:sp])
                    
                    if embtoks is not None:
                        embtoks = np.concatenate((embtoks,results), axis = 0)
                        #for i in range(len(results)):
                        #    embtoks[i] = np.concatenate((embtoks[i][:len(embtoks[i])-1],results[i][1:]),axis=0)

                    else:
                        embtoks = results
                #merge embtoks?

            else:
                embtoks = esm_embed(seq)

            reduced_embedding = embtoks.mean(0)
            batch_embeddings = np.vstack([batch_embeddings, reduced_embedding])

        #batch_embeddings = reduced_embeddings
        batch_ids = np.array([seq.name for seq in sequences], dtype="S100")
        
        # Append to file
        with h5py.File(full_out_path, "a") as f:
            dset_embed = f["embeddings"]
            dset_ids = f["ids"]
            end_idx = start_idx + batch_embeddings.shape[0]
            # Resize datasets to accommodate new data    
            dset_embed.resize((end_idx, dim))
            dset_ids.resize((end_idx,))
            # Write the new data
            dset_embed[start_idx:end_idx] = batch_embeddings
            dset_ids[start_idx:end_idx] = batch_ids
        batch_embeddings = None
        batch_ids = None
        
    return full_out_path


if __name__ == '__main__':

    fasta_path = sys.argv[1]
    out_name = sys.argv[2]
    
    get_esm_embeddings_to_hd5(fasta_path,out_name,dir = False)