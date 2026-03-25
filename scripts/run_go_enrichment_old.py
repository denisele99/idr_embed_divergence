
#Parameters:

# Type of enrichment: NN, BLAST, random, 
# k = 50 default
# alpha value for enrichment filtering: default = 0.01

#Read from config file OR commmand line?


#Dataset should be HUMAN embeddings

import pandas as pd
import sys
import argparse
sys.path.append('../src/')

from go_enrichment.go_enrichment_old import DistanceMatrix,go_enrichment_BLAST,go_enrichment_random
from utils.helpers import load_config

CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
config_data = load_config(CONFIG_PATH)

model = 'NN' #default model
dataset_path = config_data.get("target_embeddings")
k= config_data.get("k")  #default k=50
output_path = '/home/moseslab/denise/Paper/res/go_enrichment/go_enrichment_50nn_results.csv'

parser = argparse.ArgumentParser(description='Run GO enrichment for NN embeddings.')
parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file.')
parser.add_argument('--search_ids', type=str, required=False, help='Query IDs to search. (id1,id2,...). Else use all IDs in dataset.')
parser.add_argument('--k', type=int, default=50, help='Number of nearest neighbors to consider.')
parser.add_argument('--output_path', type=str, required=True, help='Directory to save the output files.')
parser.add_argument('--model', type=str, default='NN', required=True, help='Directory to save the output files.')
parser.add_argument('--alpha', type=str, default='0.01', required=False, help='Alpha value to filter significantly enriched GO terms.')

#Read arguments from config file?

def main():
    args = parser.parse_args()
    
    if args.model == 'NN':
        #Run GO enrichment of k-NN for each search id
        embed_df=pd.read_csv(args.dataset, header=None)
        enrich = DistanceMatrix(embed_df)
        
        if args.search_ids:
            args.search_ids = args.search_ids.split(',')
            query_ids = args.search_ids
        else:
            query_ids = list(enrich.distance_ids)
            
        enrich_df = enrich.run_go_enrichment_dataframe(query_ids=query_ids, k=args.k,alpha=float(args.alpha))
        
        #write output to CSV
        enrich_df.to_csv(args.output_path, index=True)
        
    elif args.model == 'BLAST':
        #Enrichment from BLAST results table default TODO add function used for BLAST search in valr
        go_enrichment_BLAST(blast_res_path=args.dataset, k=args.k, alpha=float(args.alpha), outpath=args.output_path)

    elif args.model == 'RANDOM':
        #Enrichment from random sampling of background genes
        embed_df = pd.read_csv(args.dataset, header=None)
        all_ids = list(embed_df[0])
        
        if args.search_ids:
            args.search_ids = args.search_ids.split(',')
            query_ids = args.search_ids
        
        else:
            query_ids = all_ids
        
        enrich_df = go_enrichment_random(query_ids=query_ids, all_ids=all_ids, k=args.k, alpha=float(args.alpha))
        enrich_df.to_csv(args.output_dir, index=False)

    else:
        print(f"Model {model} not recognized. Please use 'NN', 'BLAST', or 'RANDOM'.")

if __name__ == "__main__":
    main()

#go enrichment from top k BLAST hits (different requirements)
#if results file doesn't exist, run BLAST first?
