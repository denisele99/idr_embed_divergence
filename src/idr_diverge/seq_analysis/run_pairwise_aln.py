
#Edited 04-01-2025

# This script is used to run pairwise alignment of sequences using MAFFT, where input is a list of pairs and a fasta file
#Made to be run in parallel from command line

import sys
import subprocess
import os
import pandas as pd
import pickle
import argparse

#from IDR_LM.src.seq_analysis.aln_pipeline2 import write_fasta, read_aln_fasta, write_pickle, open_pkl
sys.path.append('/home/moseslab/denise/IDR_LM/src/general')
from helper_functions import read_fasta
from aln_pipeline2 import write_fasta, read_aln_fasta, write_pickle, open_pkl

#Default arguments:
perl_script = '/home/moseslab/denise/IDR_LM/src/seq_analysis/align_mafft.pl'
OUTDIR = '/home/moseslab/denise/IDR_LM/src/seq_analysis/RES'



def get_pairwise_alignments(pair_name,seq_dict,out_path='./test.csv'):#,score_dict = blosum_scores_dict): #input is list/pair of records
    print(pair_name)
    gene1_id, gene2_id = pair_name.split('-')
    #print(gene1_id,gene2_id)

    seq_dict_pair = {gene1_id:seq_dict[gene1_id], gene2_id:seq_dict[gene2_id]}

    #Write pair to fasta file
    print('Write pair to fasta file...')
    pair_fasta_path = f'{OUTDIR}/{gene1_id}_{gene2_id}.fasta'
    write_fasta(pair_fasta_path, seq_dict_pair)
    
    #Run mafft alignment
    print('Running MAFFT...')
    subprocess.run(['perl',perl_script,pair_fasta_path])
    
    #Read alignment
    print('Read and append alignment...')
    aln_path = f'{OUTDIR}/{gene1_id}_{gene2_id}.fas.aln'
    alignment = read_aln_fasta(aln_path)
    
    
    #APPEND to specified file
    if os.path.isfile(out_path): #if path exists:
        #append
        row = pd.DataFrame({'pair':[f'{gene1_id},{gene2_id}'],'gene1': [alignment[gene1_id]],'gene2': [alignment[gene2_id]]})
        row.to_csv(out_path,mode='a',header=False,index=False)
    else:
        row = pd.DataFrame({'pair':[f'{gene1_id},{gene2_id}'],'gene1': [alignment[gene1_id]],'gene2': [alignment[gene2_id]]})
        row.to_csv(out_path,mode='w',header=True,index=False)

    #Remove fasta and aln file
    print(f'Removing files {pair_fasta_path}, {aln_path}')
    subprocess.run(['rm',pair_fasta_path])
    subprocess.run(['rm',aln_path])

def main():
    
    parser = argparse.ArgumentParser(description="Run pairwise alignments.")
    parser.add_argument("--FASTA", required=True, help="Fasta path.")
    parser.add_argument("--OUTPATH", required=True, help="Output directory to write alignment results.")
    parser.add_argument("--PAIRS", required=True, nargs='+',help="Gene pairs: Format= gene1-gene2;gene3-gene4;etc.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing them.")

    args = parser.parse_args()
    fasta_dict = read_fasta(args.FASTA)
    
    for pair in args.PAIRS:#.split(';'):
        #print(pair)
        get_pairwise_alignments(pair, fasta_dict, args.OUTPATH)

    # Optionally, you can write the final output to a pickle file
    # write_pickle(OUTPATH.replace('.csv', '.pkl'), fasta_dict)
if __name__ == "__main__":
    main()



# import os
# import subprocess
# from itertools import combinations
# import time
# import sys
# sys.path.append('/home/moseslab/denise/IDR_LM/src/general')
# from helper_functions import read_fasta

# cdk_list = 'O94921 P06493 P11802 P24941 P49336 P50613 P50750 Q00526 Q00534 Q00535 Q00536 Q00537 Q07002 Q14004 Q15131 Q8IZL9 Q96Q40 Q9BWU1 Q9NYV4 Q9UQ88 P21127'.split(' ')
# fasta_path = '/home/moseslab/denise/IDR_LM/data/sequences/IDRs/human_idrs/updated/CDK_IDRs_orthologs_11-05.fasta'
# out_path= '/home/moseslab/denise/IDR_LM/src/experiments/results/CDK_IDRs_orthologs_11-05_aln.csv'

# fasta_dict =read_fasta(fasta_path)

# genes_list = cdk_list
# #selected_ids =[]
# for gene in genes_list:
#     print(f'GENE: {gene}')
#     gene_ids = [key for key in fasta_dict.keys() if gene in key]
#     if len(gene_ids)>1:
#         selected_ids = gene_ids
        
#     all_pair_combinations = []
#     # Generate all pairwise combinations of IDs
#     #for ids in selected_ids:
#     pair_combinations = [f'{a},{b}' for a,b in combinations(selected_ids,2)]
#     all_pair_combinations.extend(pair_combinations)
    
#     for pair in all_pair_combinations : #pair may contain multiple pairs
#         #print(pair)
#         get_pairwise_alignments(pair, fasta_dict, out_path)