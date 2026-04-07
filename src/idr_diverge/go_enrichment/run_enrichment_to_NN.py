#For Figure 1c

import pandas as pd
import numpy as np
from collections import defaultdict

from pygosemsim import download, graph
from pygosemsim import similarity

from idr_diverge.distances.compute_ndist import _load_embeddings
from idr_diverge.distances.embed_distance import EmbedDistanceMatrix
from idr_diverge.go_enrichment.go_enrich_eval import go_bma_similarity, go_jaccard_similarity


# def main():
#     #Look at GO enrichments of different bins of NNs
#     emb_path = '/home/moseslab/denise/embeddings/RES/human_idrs_pos_updated_esm.csv'
#     human_idr_embed = pd.read_csv(emb_path,header=None)
#     human_idr_embed[0] = human_idr_embed[0].apply(lambda x: x[x.find('_')+1:])
    
#     enrichment_df = pd.read_csv('/home/moseslab/denise/IDR_LM/results/go_enrichments/IDR_go_enrichments_esm.csv')
#     enrichment_df['Significantly Enriched GO terms'] = enrichment_df['Significantly Enriched GO terms'].apply(lambda x: eval(x))
#     enrichment_dict = enrichment_df.set_index('Unnamed: 0')['Significantly Enriched GO terms'].to_dict()

#     distmatrix= DistanceMatrix(human_idr_embed)

#     query_ids = distmatrix.distance_ids
#     ranges = [(1,2),(2, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)]
    
#     #annot_df = pd.read_table('/home/moseslab/denise/IDR_LM/data/annotations/idr_annotations/uniprotkb_Human_AND_model_organism_9606_2024_08_20.tsv')
#     #go_annotation_dict = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))

#     range_dict = defaultdict(dict)#{f'{start}-{end}':[] for start,end in ranges}

    
    
#     #G = graph.from_resource("/home/moseslab/denise/IDR_LM/data/annotations/go_2024")
#     #similarity.precalc_lower_bounds(G)
    
    

#     for query in query_ids:
#         print(query)
#         go_query = enrichment_dict.get(query, [])
        
#         if len(go_query)<1:
#             continue
        
        
#         query_idx = id_to_idx.get(query)
#         if idx is None:
#             continue

#         row = dm[query_idx]
#         neighbours = np.argpartition(row, k + 1)[:k + 1]
#         neighbours = neighbours[neighbours != query_idx][:k]
        
#         sorted_idx = np.argsort(row, axis=1)
        
#         nn_50_100 = sorted_idx[query_idx][1:][49:100]
        
#         id_idx = np.where(np.array(distmatrix.distance_ids) == query)[0][0]
#         neighbors = distmatrix.sorted_distance_m[id_idx]
#         neighbors = neighbors[neighbors != id_idx]
#         NN_ids = [distmatrix.distance_ids[idx] for idx in neighbors]
#         #NN_ids = [id.split('_')[0] for id in NN_ids]
        
#         range_dict[query] = {}
        
        
#         for start, end in ranges:
#             #print(start,end)
#             NN_ids_temp = NN_ids[start:end]
            
#             NN_GO_terms =[enrichment_dict.get(id, []) for id in NN_ids_temp]
#             NN_GO_terms = [go for go in NN_GO_terms if len(go)>0]
            
#             GO_sim = [go_sim_sem_score2(G,go_query, go_NN) for go_NN in NN_GO_terms]
#             range_dict[query][f'{start}-{end}']=np.mean(GO_sim)
        
#         df_temp = pd.DataFrame.from_dict({query: range_dict[query]}, orient='index')
#         df_temp.to_csv('/home/moseslab/denise/IDR_LM/results/go_enrichments/real_GO_enrichment_semsim_global_02-25.csv', mode='a', header=False)
from functools import lru_cache

def normalize_go(go_terms):
    return frozenset(go_terms)

@lru_cache(maxsize=None)
def cached_go_bma_similarity(go_query_t, go_nn_t):
    return go_bma_similarity(list(go_query_t), list(go_nn_t))

def main():
    
    enrich_path='/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_goa_idr_bg_nodup_a01_03_13_EVAL.csv'
    embed_path = '/home/moseslab/denise/Paper/data/embed/human_idrs_esm1b.csv'
    out_path = '/home/moseslab/denise/Paper/res/go_enrich/increasing_NN_mean_enrichment_bmasim.csv'
    
    embed_df = _load_embeddings(embed_path)
    emb_matrix = EmbedDistanceMatrix(embed_df)
    dm = emb_matrix.distance_matrix
    all_ids = emb_matrix.ids
    id_to_idx = {id_: k for k, id_ in enumerate(all_ids)}
    
    query_ids = all_ids
    
    ranges = [(1,2),(2, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)]
    max_end = max(end for _, end in ranges)
    range_dict = defaultdict(dict)
    
    enrich_df = pd.read_csv(enrich_path)
    enrich_dict = enrich_df.set_index('query_id')['term_adj_p'].apply(lambda x: list(eval(x).keys())).to_dict()
    go_terms_by_idx = [enrich_dict.get(id_, []) for id_ in all_ids]
    
    
    for query in query_ids:
        print(query)
        #go_query = enrich_dict.get(query, [])
        query_idx = id_to_idx.get(query)
        
        if query_idx is None:
            print(f'Skipping {query}... idx does not match embed_path')
            continue
        
        go_query = go_terms_by_idx[query_idx]
        
        if not go_query:
            print(f'Skipping {query}.. go enrichments not found')
            continue
        
        go_query_t = normalize_go(go_query)
        
        row = dm[query_idx]
        neighbours = np.argpartition(row, max_end + 1)[:max_end + 1]
        neighbours = neighbours[neighbours != query_idx]
        neighbours = neighbours[np.argsort(row[neighbours])] #neighbours sorted
        
        
        
        range_dict[query] = {}
        
        for start, end in ranges:
            
            nn_range = neighbours[start-1:end]
            #NN_ids = [all_ids[idx] for idx in  NN_range]
            
            nn_go_terms = [go_terms_by_idx[idx] for idx in nn_range if go_terms_by_idx[idx]]
            #NN_go_terms = [enrich_dict.get(id, []) for id in NN_ids]
            #NN_go_terms = [go for go in NN_go_terms if len(go)>0]
            
            if not nn_go_terms:
                print(f'No go terms found for neighbours in range {start}-{end}')
                range_dict[query][f'{start}-{end}']=np.nan
                continue
            
            
            #go_sim = [go_bma_similarity(go_query, go_nn) for go_nn in nn_go_terms]
            go_sim = [cached_go_bma_similarity(go_query_t,normalize_go(go_nn)) for go_nn in nn_go_terms]
            range_dict[query][f'{start}-{end}']=np.mean(go_sim) if go_sim else np.nan
            
        df_temp = pd.DataFrame.from_dict({query: range_dict[query]}, orient='index')
        df_temp.to_csv(out_path, mode='a', header=False)

def main1():
    
    enrich_path='/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_goa_idr_bg_nodup_a01_03_13_EVAL.csv'
    embed_path = '/home/moseslab/denise/Paper/data/embed/human_idrs_esm1b.csv'
    out_path = '/home/moseslab/denise/Paper/res/go_enrich/increasing_NN_mean_enrichment_jaccsim_03_26.csv'
    
    embed_df = _load_embeddings(embed_path)
    emb_matrix = EmbedDistanceMatrix(embed_df)
    dm = emb_matrix.distance_matrix
    all_ids = emb_matrix.ids
    id_to_idx = {id_: k for k, id_ in enumerate(all_ids)}
    
    query_ids = all_ids
    
    ranges = [(1,2),(2, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 10000)]
    max_end = max(end for _, end in ranges)
    range_dict = defaultdict(dict)
    
    enrich_df = pd.read_csv(enrich_path)
    enrich_dict = enrich_df.set_index('query_id')['term_adj_p'].apply(lambda x: list(eval(x).keys())).to_dict()
    go_terms_by_idx = [enrich_dict.get(id_, []) for id_ in all_ids]
    
    
    for query in query_ids:
        print(query)
        #go_query = enrich_dict.get(query, [])
        query_idx = id_to_idx.get(query)
        
        if query_idx is None:
            print(f'Skipping {query}... idx does not match embed_path')
            continue
        
        go_query = go_terms_by_idx[query_idx]
        
        if not go_query:
            print(f'Skipping {query}.. go enrichments not found')
            continue
        
        #go_query_t = normalize_go(go_query)
        
        row = dm[query_idx]
        neighbours = np.argpartition(row, max_end + 1)[:max_end + 1]
        neighbours = neighbours[neighbours != query_idx]
        neighbours = neighbours[np.argsort(row[neighbours])] #neighbours sorted
        
        
        
        range_dict[query] = {}
        
        for start, end in ranges:
            
            nn_range = neighbours[start-1:end]
            #NN_ids = [all_ids[idx] for idx in  NN_range]
            
            nn_go_terms = [go_terms_by_idx[idx] for idx in nn_range if go_terms_by_idx[idx]]
            #NN_go_terms = [enrich_dict.get(id, []) for id in NN_ids]
            #NN_go_terms = [go for go in NN_go_terms if len(go)>0]
            
            if not nn_go_terms:
                print(f'No go terms found for neighbours in range {start}-{end}')
                range_dict[query][f'{start}-{end}']=np.nan
                continue
            
            
            go_sim = [go_jaccard_similarity(go_query, go_nn) for go_nn in nn_go_terms]
            #[go_bma_similarity(go_query, go_nn) for go_nn in nn_go_terms]
            #go_sim = [cached_go_bma_similarity(go_query_t,normalize_go(go_nn)) for go_nn in nn_go_terms]
            range_dict[query][f'{start}-{end}']=np.mean(go_sim) if go_sim else np.nan
            
        df_temp = pd.DataFrame.from_dict({query: range_dict[query]}, orient='index')
        df_temp.to_csv(out_path, mode='a', header=False)

def main2():

    enrich_path = '/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_goa_idr_bg_nodup_a01_03_13_EVAL.csv'
    embed_path = '/home/moseslab/denise/Paper/data/embed/human_idrs_esm1b.csv'
    out_path = '/home/moseslab/denise/Paper/res/go_enrich/increasing_NN_mean_enrichment_bmasim_03_26.csv'

    embed_df = _load_embeddings(embed_path)
    emb_matrix = EmbedDistanceMatrix(embed_df)
    dm = emb_matrix.distance_matrix
    all_ids = emb_matrix.ids
    id_to_idx = {id_: k for k, id_ in enumerate(all_ids)}

    ranges = [(1,2),(2, 50), (51, 100), (101, 200), (201, 500), (501, 1000),
              (1001, 2000), (2001, 5000), (5001, 10000)]
    max_end = max(end for _, end in ranges)

    enrich_df = pd.read_csv(enrich_path)
    enrich_dict = enrich_df.set_index('query_id')['term_adj_p'].apply(lambda x: list(eval(x).keys())).to_dict()

    go_terms_by_idx = [enrich_dict.get(id_, []) for id_ in all_ids]
    normalized_go_by_idx = [tuple(normalize_go(go)) if go else tuple() for go in go_terms_by_idx]
    
    
    df_ranges = pd.read_csv('/home/moseslab/denise/Paper/res/go_enrich/increasing_NN_mean_enrichment_sim_03_20.csv',header=None)
    current_ids = list(df_ranges[0])
    df_ranges = None
    
    enriched_queries = list(enrich_df[enrich_df['terms'].map(eval).str.len() >0]['query_id'])
    remaining_ids = list(set(enriched_queries).symmetric_difference(set(current_ids)))
    enrich_df = None
    results = []
    batch_size = 10
    buffer = []
    
    write_header= True

    for query in remaining_ids:
        print(query)
        query_idx = id_to_idx.get(query)
        go_query_t = normalized_go_by_idx[query_idx]
        
        if not go_query_t:
            print(f"Skipping {query}.. go enrichments not found")
            continue
        
        #print('Finding 50 NN')
        row = dm[query_idx]
        neighbours = np.argpartition(row, max_end + 1)[:max_end + 1]
        neighbours = neighbours[neighbours != query_idx]
        neighbours = neighbours[np.argsort(row[neighbours])]

        #print('Calculating Go similarity')
        sim_scores = []
        for idx in neighbours[:max_end]:
            go_nn_t = normalized_go_by_idx[idx]
            if go_nn_t:
                sim_scores.append(cached_go_bma_similarity(go_query_t, go_nn_t))
            else:
                sim_scores.append(np.nan)

        sim_scores = np.array(sim_scores, dtype=float)

        row_result = {"query_id": query}
        for start, end in ranges:
            vals = sim_scores[start-1:end]
            vals = vals[~np.isnan(vals)]
            row_result[f"{start}-{end}"] = vals.mean() if len(vals) else np.nan

        buffer.append(row_result)
        
        if len(buffer) >= batch_size:
            df_batch = pd.DataFrame(buffer)
            df_batch.to_csv(
                out_path,
                mode="a",
                header=write_header,
                index=False
            )
            buffer.clear()
            write_header = False
        

    #pd.DataFrame(results).to_csv(out_path, index=False)
    if buffer:
        df_batch = pd.DataFrame(buffer)
        df_batch.to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )
        

if __name__ == '__main__':
    main2()