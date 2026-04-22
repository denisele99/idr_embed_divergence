#Copied from /home/moseslab/denise/IDR_LM/src/experiments/final_scripts/calc_go_enrichment.py 06/27/25

import ast
import sys
from collections import Counter
import numpy as np
import pandas as pd
import os
import csv
# from itertools import combinations

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

#sys.path.append('/home/moseslab/denise/IDR_LM/src/analysis')
from distances.embed_distance import EmbedDistanceMatrix

from .go_enrichment_analysis_old import load_go_annotations
from utils.helpers import load_config


CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
config_data = load_config(CONFIG_PATH)

go_annotation_dict = load_go_annotations(config_data.get("go_annotations"))


def go_enrichment(study_ids, go_terms_background, gene_id_background, filter_alpha=0.05,go_annotation_dict = go_annotation_dict):
    """
    Perform GO enrichment analysis using Fisher's Exact Test and multiple testing correction.
    
    Parameters:
        study_ids (list): List of IDs in the study set (e.g., nearest neighbors).
        go_terms_background (dict): Dictionary mapping gene IDs to their GO terms for the background population.
        gene_id_background (list): List of all gene IDs in the background population.
        filter_alpha (float): Significance level to filter enriched terms after multiple testing correction.
    
    Returns:
        dict: Corrected results mapping GO terms to their adjusted p-values (significant terms if filter_alpha is provided).
    """
    # Extract GO terms for study IDs
    study_go_terms = [go_annotation_dict.get(id, []) for id in study_ids]
    study_go_terms_flat = [term for sublist in study_go_terms for term in sublist]
    study_go_counts = Counter(study_go_terms_flat)

    # Prepare a copy of the background GO terms and remove study IDs
    go_background_temp = go_terms_background.copy()
    #for key in study_ids:
    #    go_background_temp.pop(key, None)

    # Count GO term occurrences in the background
    background_go_counts = Counter(
        [term for sublist in go_background_temp.values() for term in sublist]
    )

    # Collect all unique GO terms from both study and background sets
    #all_go_terms = set(study_go_counts) | set(background_go_counts)
    all_go_terms = set(study_go_counts) #UPDATED: only study GO terms

    
    # Perform Fisher's Exact Test for each GO term
    results = []
    for go in set(study_go_terms_flat): #modified from all_go_terms
        study_count = study_go_counts.get(go, 0)
        background_count = background_go_counts.get(go, 0)
        
        table = [
            [study_count, len(study_ids) - study_count],
            [background_count, len(gene_id_background) - background_count]
        ]
        _, p_value = fisher_exact(table)
        results.append((go, p_value))
    
    if not results:
        return {}, {}

    # Adjust p-values for multiple testing using FDR correction
    p_values = np.array([p for _, p in results])
    adjusted_p_values = multipletests(p_values, alpha=filter_alpha, method='fdr_bh')[1]
    corrected_results = {
        go: adj_p for (go, _), adj_p in zip(results, adjusted_p_values)
    }

    # Filter GO terms based on the significance threshold
    if filter_alpha:
        corrected_results = {k: v for k, v in corrected_results.items() if v < filter_alpha}
    
    #Get counts
    go_enrichment_counts= {
                go: f"{study_go_counts[go]}/{background_go_counts[go]}"
                for go in corrected_results.keys()
            }

    return corrected_results,go_enrichment_counts


class DistanceMatrix:
    """
    A class for computing distance matrices, finding k-nearest neighbors (k-NN),
    and performing GO enrichment analysis for embedding-based data.
    """

    def __init__(self, embed_df):
        # self.emblookup = EmbeddingLookup(embed_df)
        # self.distance_m, self.distance_ids = self.emblookup.run_embedding_lookup_distance(
        #     querys=self.emblookup.embedding_db, metric="cosine"
        # )
        
        self.emblookup = EmbedDistanceMatrix(embed_df) #TODO test to see if this still works
        self.distance_m, self.distance_ids = self.emblookup.run_embedding_lookup_distance(
            queries=self.emblookup.embedding_db, metric="cosine"
        )
        
        
        self.sorted_distance_m = np.argsort(self.distance_m, axis=1)
        self.go_annotation_dict = go_annotation_dict
        self.go_terms_background = {id.split("_")[0]: self.go_annotation_dict.get(id.split("_")[0], []) for id in self.distance_ids}

    def find_kNN(self, query_ids: list, k=50):
        """fro
        Find the k-nearest neighbors (k-NN) for a list of query IDs.

        Args:
            query_ids (list): List of query IDs to find neighbors for.
            k (int): Number of nearest neighbors to retrieve (default=50).

        Returns:
            dict: Dictionary mapping query IDs to their k-nearest neighbors.
        """
        ids = np.array(self.distance_ids)
        kNN_dict = {}

        for query in query_ids:
            id_idx = np.where(ids == query)[0][0]
            neighbors = self.sorted_distance_m[id_idx]
            neighbors = neighbors[neighbors != id_idx][:k]
            NN_ids = [ids[idx].split("_")[0] for idx in neighbors]
            kNN_dict[query] = NN_ids

        return kNN_dict

    def go_enrichment_kNN(self, query_ids, k=50, alpha=0.05):
        """
        Perform GO term enrichment analysis for k-NN groups.

        Args:
            query_ids (list): List of query IDs for enrichment analysis.
            go_annotation_dict (dict): Dictionary mapping gene IDs to GO terms.
            NN (int): Number of nearest neighbors to consider (default=50).
            alpha (float): Significance level for enrichment (default=0.05).

        Returns:
            tuple: (go_enrichment_dict, go_enrichment_counts)
        """
        #ids = self.distance_ids
        #go_background = {id.split("_")[0]: go_annotation_dict.get(id.split("_")[0], []) for id in ids}
        go_background = self.go_terms_background
        kNN_dict = self.find_kNN(query_ids, k=k)

        go_enrichment_dict = {}
        go_enrichment_counts = {}

        for query, NN_ids in kNN_dict.items():
            
            NN_ids_short =  [id.split('_')[0] for id in NN_ids]
            enriched_goterms,enriched_gocounts = go_enrichment(study_ids=NN_ids_short,
                          go_terms_background=go_background,
                          gene_id_background=list(go_background.keys()),
                          filter_alpha=alpha)
            
            #go_enrichment_dict[query] = list(enriched_goterms.keys())
            go_enrichment_dict[query]  = list(enriched_goterms.keys())
            go_enrichment_counts[query] = enriched_gocounts
            #go_enrichment_counts[query] = {
            #    go: f"{NN_count[go]}/{population_counts[go]}"
            #    for go in enriched_goterms.keys()
            #}

        return go_enrichment_dict, go_enrichment_counts

    def run_go_enrichment_dataframe(self, query_ids, k=50, alpha=0.05, type="counts", geneid_dict=None):
        #TODO add option to save/append to a dataframe csv
        """
        Run GO enrichment and return results as a DataFrame.
        Convert GO enrichment results into a DataFrame.

        Args:
            query_ids (list): List of query IDs for enrichment.
            NN (int): Number of nearest neighbors (default=50).
            alpha (float): Significance level for enrichment (default=0.05).
            type (str): Output format ('counts' or 'function').
            geneid_dict (dict, optional): Mapping of gene IDs for annotations.

        Returns:
            pd.DataFrame: DataFrame containing GO enrichment results.
        """
        #enrichment dict with counts
        #go_enrichment_counts is a dict of dicts
        
        go_enrichment_dict,go_enrichment_counts = self.go_enrichment_kNN(query_ids, k=k, alpha=alpha)
        
        df_enrichment = pd.DataFrame({'Significantly Enriched GO terms': go_enrichment_dict, 'GO term counts': go_enrichment_counts})

        if geneid_dict:
            df_enrichment.index = df_enrichment.index.map(lambda x: geneid_dict.get(x.split("_")[0], x))
        
        if type == "function":
            df_enrichment['Significantly Enriched GO terms'] = df_enrichment['Significantly Enriched GO terms'].apply(lambda x: go_ID_to_function([go])[go] for go in x)
        
        return df_enrichment


def go_enrichment_BLAST(blast_res_path,k=50,alpha=0.05,outpath='out.csv', go_annotation_dict=go_annotation_dict): #TODO check if this works
    blast_res = pd.read_table(blast_res_path, header=None)
    blast_res[0] = blast_res[0].apply(lambda x: x[x.find('_')+1:])
    blast_res[1] = blast_res[1].apply(lambda x: x[x.find('_')+1:])

    query_ids = list(set(blast_res[0]))
    
    # Get background go term counts
    gene_id_background = [id.split('_')[0] for id in query_ids]
    go_background = {id: go_annotation_dict.get(id, []) for id in gene_id_background}

    # Initialize the CSV file with headers
    if not(os.path.exists(outpath)):
        with open(outpath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Query', 'Significantly Enriched GO terms', 'GO term counts']) #Only write headers once

    for query in query_ids:
        NN_ids = list(blast_res[blast_res[0] == query].iloc[:, 1])
        query_name = query.split('_')[0]
        NN_ids_short = [id.split('_')[0] for id in NN_ids if not(query_name in id)][:k]  # 50 NN

        enriched_goterms, enriched_gocounts = go_enrichment(
            study_ids=NN_ids_short,
            go_terms_background=go_background,
            gene_id_background=list(go_background.keys()),
            filter_alpha=alpha
        )
        #go_enrichment_dict[query] = list(enriched_goterms.keys())
        # Append the results to the CSV file
        with open(outpath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([query, list(enriched_goterms.keys()), enriched_gocounts])
    
    #return go_enrichment_dict


def go_enrichment_random(query_ids, all_ids,k=50,alpha=0.05):
    """
    Perform GO enrichment analysis by randomly sampling 
    a subset of background gene IDs and comparing their GO term distributions for n_samples times.
    """
    #dist_matrix = DistanceMatrix(embed_df=embed_df)
    ids = np.array(all_ids) #np.array(list(embed_df[0]))
    go_background = {id.split("_")[0]: go_annotation_dict.get(id.split("_")[0], []) for id in ids}
    go_enrichment_dict = {}
    
    for query in query_ids:
        NN_idx = np.random.choice(len(ids), size=k, replace=False)
        NN_ids = ids[NN_idx]
        NN_ids_short= [id.split('_')[0] for id in NN_ids] #Only use the gene names
        
        enriched_goterms,_ = go_enrichment(study_ids=NN_ids_short,
                            go_terms_background=go_background,
                            gene_id_background=list(go_background.keys()),
                            filter_alpha=alpha)

        go_enrichment_dict[query] = list(enriched_goterms.keys())
    
    
    df = pd.DataFrame.from_dict(go_enrichment_dict,orient='index')
    #df.to_csv(outpath)

    return df


