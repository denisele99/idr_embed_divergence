import os
import sys
from itertools import combinations
import subprocess
#from Bio.SubsMat import MatrixInfo
import pandas as pd
import time
from multiprocessing import Pool
import pickle
#sys.path.append('/valr/denise/Embed_Distance/src')
#from seq_analysis.calc_blosum import score_pairwise
sys.path.append('/home/moseslab/denise/IDR_LM/src/general')
from helper_functions import read_fasta



#Default paths 
perl_script = './align_mafft.pl'
out_dir = './res/'


#import paths
#matrix = MatrixInfo.blosum62


#Define helper functions
def write_fasta(filename,sequences:dict):
    with open(filename, 'w') as file:
        for name, sequence in sequences.items():
            file.write(f">{name}\n{sequence}\n")

def read_aln_fasta(file):
    fasta_dict = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                seq_id = line[1:]
                if seq_id not in fasta_dict:
                    fasta_dict[seq_id] = []
                continue
            sequence = line
            fasta_dict[seq_id].append(sequence)
    for seq_id, seq_list in fasta_dict.items():
        fasta_dict[seq_id] = ''.join(seq_list)
    return fasta_dict

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def open_pkl(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)
    return loaded_file



#Running functions
def get_pairwise_alignments(pair_name,seq_dict,out_path='./test.csv'):#,score_dict = blosum_scores_dict): #input is list/pair of records
    print(pair_name)
    gene1_id = pair_name[0]
    gene2_id = pair_name[1]

    seq_dict_pair = {gene1_id:seq_dict[gene1_id], gene2_id:seq_dict[gene2_id]}

    #Write pair to fasta file
    name = out_dir +  gene1_id +'_'+gene2_id
    pair_fasta_path= name+'.fasta'

    write_fasta(pair_fasta_path, seq_dict_pair)
    
    #Run mafft alignment
    subprocess.run(['perl',perl_script,pair_fasta_path])
    
    #Read alignment
    aln_path = name +'.fas.aln'
    alignment = read_aln_fasta(aln_path)
    
    
    #APPEND to specified file
    if os.path.isfile(out_path): #if path exists:
        #append
        row = pd.DataFrame({'pair':[gene1_id +'_'+gene2_id],'gene1': [alignment[gene1_id]],'gene2': [alignment[gene2_id]]})
        row.to_csv(out_path,mode='a',header=False,index=False)
    else:
        row = pd.DataFrame({'pair':[gene1_id +'_'+gene2_id],'gene1': [alignment[gene1_id]],'gene2': [alignment[gene2_id]]})
        row.to_csv(out_path,mode='w',header=True,index=False)

    #Remove fasta and aln file
    subprocess.run(['rm',pair_fasta_path])
    subprocess.run(['rm',aln_path])
    


def write_pairwise_aln_run(input_fasta_path,out_dir,out_name ='test'):
    '''Write all by all pairwise alignments from a fasta file to csv, then pickle'''
    
    start_time = time.time()
    
    input_file = read_fasta(input_fasta_path) 
    seq_dict = {rec.id: str(rec.seq) for rec in input_file}
    pair_combinations = [[a,b] for a,b in combinations(seq_dict.keys(),2)]
    out_path = out_dir+out_name
    
    with Pool(50) as pool:
        pool.starmap(get_pairwise_alignments, [(query, seq_dict,out_path+'.csv') for query in pair_combinations],chunksize=300)

    data = pd.read_csv(out_path+'.csv')
    write_pickle(out_path +'.pkl', data)
    
    #delete the other csv file
    subprocess.run(['rm',out_path+'.csv'])    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    


# def blosum_wrapper(aln1,aln2,matrix=matrix):

#     #score_averaged = score_pairwise(aln1,aln2,matrix,0,0)/len(aln1)
#     score_averaged = score_pairwise(aln1,aln2,matrix,-11, -1)/len(aln1)

#     return score_averaged

# def get_blosum_from_aln_run(out_path, df_path = '/valr/denise/codes/seq_analysis/res/all_yeast.pkl'):
#     start_time = time.time()
    
#     df = open_pkl(df_path) #Where pickle is a csv
#     records = df.to_dict('records') #df = all_yeast_df
#     names = df['pair']
#     matrix = MatrixInfo.blosum62

#     with Pool(50) as pool:
#         ##blosum_scores = pool.starmap(score_pairwise, [(rec['gene1'], rec['gene2'],matrix) for rec in records],chunksize=100)
#         #blosum_scores = pool.starmap(blosum_wrapper, [(rec['gene1'], rec['gene2'],matrix, 0,0) for rec in records],chunksize=100)
#         blosum_scores = pool.starmap(blosum_wrapper, [(rec['gene1'], rec['gene2'],matrix) for rec in records],chunksize=100)
#     blosum_dict = {name: value for name,value in zip(names,blosum_scores)}

#     #get the reverse value
#     #order = [2,3,0,1]
#     #blosum_dict_allbyall = {'_'.join([k.split('_')[i] for i in order]):v for k,v in blosum_dict.items()}
#     #blosum_dict_allbyall.update(blosum_dict)

#     #save as pkl, append to dictionary later    
#     write_pickle(out_path,blosum_dict)
    
#     print("--- %s seconds ---" % (time.time() - start_time))
    
#     #Calculate regular, nogap penalties

def seq_identity(aln1,aln2):

    # Calculate sequence identity
    sequence_identity = sum(a == b for a, b in zip(aln1, aln2)) / len(aln1)

    return sequence_identity

def get_seq_identity_from_aln_run(out_path, df_path = '/valr/denise/codes/seq_analysis/res/all_yeast.pkl'):
    start_time = time.time()
    
    df = open_pkl(df_path) #Where pickle is a csv
    records = df.to_dict('records') #df = all_yeast_df
    names = df['pair']

    with Pool(40) as pool:
        seqiden_scores = pool.starmap(seq_identity, [(rec['gene1'], rec['gene2']) for rec in records],chunksize=100)
    seqiden_dict = {name: value for name,value in zip(names,seqiden_scores)}

    #get the reverse value
    #order = [2,3,0,1]
    #blosum_dict_allbyall = {'_'.join([k.split('_')[i] for i in order]):v for k,v in blosum_dict.items()}
    #blosum_dict_allbyall.update(blosum_dict)

    #save as pkl, append to dictionary later    
    write_pickle(out_path,seqiden_dict)
    print(out_path)
    print("--- %s seconds ---" % (time.time() - start_time))

def get_seq_identity_from_aln_run2(out_path, df_path = '/valr/denise/codes/seq_analysis/res/all_yeast.pkl'):
    start_time = time.time()
    
    df = open_pkl(df_path) #Where pickle is a csv
    
    #Make input a fasta file? Or select indices of a fasta file? TODO
    
    records = df.to_dict('records') #df = all_yeast_df
    names = df['pair']

    seqiden_scores = {rec['pair']: seq_identity(rec['gene1'], rec['gene2']) for rec in records}
    seqiden_dict = {name: value for name,value in zip(names,seqiden_scores)}

    #get the reverse value
    #order = [2,3,0,1]
    #blosum_dict_allbyall = {'_'.join([k.split('_')[i] for i in order]):v for k,v in blosum_dict.items()}
    #blosum_dict_allbyall.update(blosum_dict)

    #save as pkl, append to dictionary later    
    write_pickle(out_path,seqiden_dict)
    print(out_path)
    print("--- %s seconds ---" % (time.time() - start_time))


def run_pairwise_aln_parallel(input_fasta_path, list_ids, out_path, n_processes=4):
    """
    Run pairwise alignments in parallel for a list of pairs.
    
    Parameters:
        input_fasta_path (str): Path to the input FASTA file.
        list_ids (list of lists): List of lists of ids for all by all pairwise alignment.
        out_path (str): Directory to save the output files.
    """
    processes = []
    
    all_pair_combinations = []
    # Generate all pairwise combinations of IDs
    for ids in list_ids:
        pair_combinations = [f'{a},{b}' for a,b in combinations(ids,2)]
        all_pair_combinations.extend(pair_combinations)
    
    #Split the list of pairs into chunks for parallel processing
    chunk_size = len(all_pair_combinations) // n_processes
    pair_list = [all_pair_combinations[i:i + chunk_size] for i in range(0, len(all_pair_combinations), chunk_size)]
    # Ensure the last chunk contains any remaining pairs
    if len(all_pair_combinations) % n_processes != 0:
        pair_list.append(all_pair_combinations[len(pair_list) * chunk_size:])
    
    for pairs in pair_list: #pair may contain multiple pairs
        # Construct the command
        cmd = ['python', '/home/moseslab/denise/IDR_LM/src/seq_analysis/run_pairwise_aln.py', input_fasta_path, out_path]+pairs
    
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)

    # Wait for all scripts to complete and capture their outputs
    for process in processes:
        stdout, stderr = process.communicate()
        print(f"Output of {process.args}: {stdout.decode()}")
        if stderr:
            print(f"Error in {process.args}: {stderr.decode()}")



#Run pairwise alignments of random human IDR and PFAM sequences 04/01/25
#Read from dataframe of random NN distances
def main():
    outdir= '/home/moseslab/denise/IDR_LM/src/seq_analysis/'
    CDK_random_NN_df = pd.read_csv('/home/moseslab/denise/IDR_LM/src/experiments/results/CDK_random_NN_distances_20_100.csv')
    for name in CDK_random_NN_df['Name']:
        selected_ids = eval(CDK_random_NN_df[CDK_random_NN_df['Name']==name]['List IDs'].values[0])
        
        #Split selected ids into chunks of 20 (20 lists)
        chunk_size = len(selected_ids) // 20
        selected_ids_chunk = [selected_ids[i:i + chunk_size] for i in range(0, len(selected_ids), chunk_size)]
        for chunk in selected_ids_chunk:
            if 'IDR' in name:
                continue
                # run_pairwise_aln_parallel(input_fasta_path='/home/moseslab/denise/IDR_LM/data/sequences/IDRs/human_idrs/human_idrs_pos_updated_labels.fasta',
                #                         list_ids=selected_ids,
                #                         out_path=f'{outdir}{name}_aln.csv')
            elif 'PFAM' in name:
                print(chunk)
                run_pairwise_aln_parallel(input_fasta_path='/home/moseslab/denise/IDR_LM/data/sequences/NON_IDRs/human_non_IDRs/human_PFAM_updated_labels.fasta',
                                        list_ids=chunk,
                                        out_path=f'{outdir}{name}_aln.csv', n_processes=1)

# main()
# selected_ids = eval(CDK_random_NN_df[CDK_random_NN_df['Name']==name]['List IDs'].values[0])
# run_pairwise_aln_parallel(input_fasta_path='/home/moseslab/denise/IDR_LM/data/sequences/NON_IDRs/human_non_IDRs/human_PFAM_updated_labels.fasta',
#                                         list_ids=chunk,
#                                         out_path=f'{outdir}{name}_aln.csv', n_processes=1)
    
#input_fasta_path = '/valr/denise/data/sequences/sample_homologs.fasta'

#Run pairwise alignments of CDK IDR and PFAM sequences

#Run pairwise alignments of simulations 



#input paths
#input_fasta_path = sys.argv[1]#'/valr/denise/data/sequences/sample_homologs.fasta'
#out_name = sys.argv[2]


#write_pairwise_aln_run(input_fasta_path,out_dir,out_name)


#get_seq_identity_from_aln_run('./allyeast_seqiden.pkl')

#Generate blosum scores again but divide by length of the protein to normalize scores?

#get_blosum_from_aln_run('/valr/denise/codes/clust_dist_analysis/dist_matrix_res/blosum_allyeast_nogapp_avg.pkl')

#Run 06/20/24
#get_blosum_from_aln_run('/valr/denise/Embed_Distance/results/seq_analysis/blosum_allyeast_gap_avg.pkl', '/neuhaus/denise/valr_data/all_yeast_aln.pkl')

