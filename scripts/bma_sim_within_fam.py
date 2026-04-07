
#Get bma similarity between enriched GO terms for each gene family
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import numpy as np
#from Paper.src.idr_diverge.go_enrichment.go_enrichment_clean import load_go_sim_graph
from idr_diverge.go_enrichment.go_enrich_eval import go_bma_similarity, go_bma_mean_similarity, load_go_sim_graph, wang_similarity
from itertools import combinations
G = load_go_sim_graph()

@lru_cache(maxsize=None)
def cached_wang_similarity(term1, term2):
    return wang_similarity(term1, term2, G)

def go_bma_similarity(go_terms1, go_terms2):
    
    if not go_terms1 or not go_terms2:
        return np.nan
     
    # Calculate similarity using BMA method
    similarity_scores = []
    for term1 in go_terms1:
        try:
            best_match_score = max(wang_similarity(term1, term2, G) for term2 in go_terms2)
            similarity_scores.append(best_match_score)
        except:
            continue

    for term2 in go_terms2:
        try:
            best_match_score = max(wang_similarity(term2, term1, G) for term1 in go_terms1)
            similarity_scores.append(best_match_score)
        except:
            continue
    
    bma_similarity = sum(similarity_scores) / (len(go_terms1) + len(go_terms2))
    
    return bma_similarity

def go_bma_mean_similarity(go_terms_list):
    '''
    BMA mean similarity across list of go term lists (all pairs vs all pairs)
    
    :param go_terms_list: list of lists of go terms
    :return: mean BMA similarity
    '''
    #go_terms_list = list(set([term for sublist in go_terms_list for term in sublist]))
    if len(go_terms_list) <2:
        return np.nan
    else:
        sims = []
        for terms1,terms2 in combinations(go_terms_list, 2):
            #print(terms1, terms2)
            if len(terms1)>0 and len(terms2)>0:
                sims.append(go_bma_similarity(terms1, terms2))
        #print(sims)
        mean_similarity = np.mean(sims)
        return mean_similarity


# def get_enrichment_terms(uniprot_ids, go_enrich_dict):
#     go_terms_list = [terms for key, terms in go_enrich_dict.items() if len(terms)>0 and any(key.startswith(uid) for uid in uniprot_ids)]
#     return go_terms_list
def get_enrichment_terms(uniprot_ids, uid_to_terms):
    return [uid_to_terms.get(uid, []) for uid in uniprot_ids]

enrich_df = pd.read_csv('/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_goa_idrs_a01_04_01_EVAL.csv') #'/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_goa_idr_bg_nodup_a01_03_13_EVAL.csv')
enrich_df['terms'] = enrich_df['terms'].map(eval)
enrich_df.set_index('query_id', inplace=True)
go_enrich_dict = enrich_df['terms'].to_dict()

uid_to_terms = defaultdict(list)
for key, terms in go_enrich_dict.items():
    if terms:
        uid = key.split('_')[0]   # adjust if query_id format differs
        uid_to_terms[uid].extend(terms)

#print(uid_to_terms['Q9BPX3'])

data_path = '/home/moseslab/denise/Paper/res/FND/FND_results_filt_2.csv' #'/home/moseslab/denise/Paper/res/FND/FND_results_updated.csv'
data = pd.read_csv(data_path)
data.set_index('Group_id', inplace=True)

gene_group_df = pd.read_csv('/home/moseslab/denise/Paper/data/annotations/hgnc_gene_groups_01_27_26.csv')
gene_group_df.set_index('Group_id',inplace=True)
gene_group_df['uniprot_ids'] = gene_group_df['uniprot_ids'].apply(lambda x: eval(x) if pd.notnull(x) else [])
gene_group_df.rename(index={'Cyclin':'CYCLIN'}, inplace = True)

gene_group_df_filtered = gene_group_df.loc[data.index].copy()#.iloc[:10,:]
gene_group_df_filtered['bma_sim'] = gene_group_df_filtered['uniprot_ids'].apply(lambda x: 
    go_bma_mean_similarity(get_enrichment_terms(x, uid_to_terms)))

#print(gene_group_df_filtered[['uniprot_ids','bma_sim']].head())
res_combined = pd.concat([data['FND'], gene_group_df_filtered['bma_sim']], axis=1)

res_combined.to_csv('/home/moseslab/denise/Paper/res/FND/FND_filt2_within_bma_04_02.csv') #FND_random_bw_results_within_bma_04_02.csv')