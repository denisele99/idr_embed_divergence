#From /home/moseslab/denise/Thesis/src/NN_distance_analysis/scripts/compute_NN_divergence.py

#Edited 2025-09-22

from __future__ import annotations
from typing import Dict, Iterable, List, Literal, Mapping

import numpy as np
import glob
import pandas as pd
from scipy.stats import ttest_ind
import statistics as stat
from scipy import stats
import glob
import sys
import math
import random
import time
from collections import defaultdict
from itertools import product,combinations
from scipy.stats import norm
#from analyze_NN_dist import *
import numpy as np
from itertools import product
from typing import Optional


import csv
import os
from math import comb
import re
from sklearn.metrics import pairwise_distances

sys.path.append('/home/moseslab/denise/Paper/src/')

#from utils.helper_functions import save_pickle,embedding_h5_to_dataframe, get_ortholog_prefixes, find_files_by_prefix, find_files_by_identifier, extract_IDs_from_end, read_pickle
#from utils.filter_orthologs import filter_orthologs
#from config import cdk_ids, human_idr_embed_path, human_pfam_embed_path, unrelated_genes, genedict, idr_ortho_embed_dir,protein_families

#from helpers import load_config, combine_dicts, collect_values_by_key, merge_dict
from utils.helpers import combine_dicts, merge_dict, collect_values_by_key, embedding_h5_to_dataframe,save_pickle,read_pickle, find_files_by_identifier, find_files_by_prefix, extract_IDs_from_end
from distances.embed_distance import EmbedDistanceMatrix, _build_index_maps, _precompute_base_sorted, find_NN_pair_fast, compute_groupwise_NN_distances_FAST

from utils.helpers import load_config


#CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
CONFIG_PATH = '/home/moseslab/denise/Paper/src/distances/config_FND.yaml'
CONFIG_DATA = load_config(CONFIG_PATH)

#fam_dir_path='/home/moseslab/denise/Thesis/results/NN_distance/idr_families',
#unrelated_dist_path = '/home/moseslab/denise/Thesis/results/NN_distance/test/random_idrs_within_ortho_full_09_15_size100.pkl',

## HELPERS

def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _parse_ids(ortholog_ids):
    """
    Vectorized parse of ID strings into __species and __gpos columns.
    Assumes IDs look like 'SPECIES_..._...'; __gpos is the join of the last |pos| tokens.
    """
    pos= -4
    ids = pd.Series(ortholog_ids).astype(str)  # Series[str]
    parts = ids.str.split('_')                      # Series[list[str]]

    k = abs(pos)
    if k == 0:
        raise ValueError("pos must be non-zero; use negative values to take last k tokens")

    # join the last k tokens (vectorized for list-like via .str.slice)
    gpos = parts.str.slice(-k).str.join('_')
    genes = parts.str.get(-k)
    species_full = parts.str.get(0)                 # first token

    # remove non-letters; if result is empty, fall back to original
    species_clean = species_full.str.replace(r'[^A-Za-z]', '', regex=True)
    species = species_clean.where(species_clean.ne(''), species_full)
    
    df = pd.DataFrame()
    df['__id'] = ids
    df['__species'] = species
    df['__ensembl_id'] = species_full
    df['__genes'] = genes
    df['__gpos'] = gpos
    return df


#Get default parameters from config file?

# def filter_orthologs(
#     ortholog_ids: Iterable[str],
#     ensembl_homol_reference: Dict = ensembl_homol_reference,
#     #percent_id_dict: Dict[str, Optional[float]],
#     ortholog_type: Literal["ortholog_one2one", "ortholog_one2many", "ortholog_one2one_one2many",] = "ortholog_one2one",
#     species_threshold: float = 0.5,
#     drop_duplicates: Literal["random", "percent_id"] = "percent_id",
#     spp_count_proteome_threshold: int = 1000,
#     n_orthologs:int=1,
#     ref=False
# ) -> List[str]:
#     """
#     Filter a batch of ortholog region IDs given homology references and a per-protein %identity map.

#     Steps
#     -----
#     1) Keep only rows whose Ensembl protein is in allowed orthologs of `ortholog_type`.
#     2) Keep only species whose total proteome count for this `ortholog_type` exceeds `spp_count_proteome_threshold`,
#        plus always keep HUMAN.
#     3) Deduplicate by (__species, __gpos):
#          - If drop_duplicates == 'percent_id': keep the row with the highest percent_id
#            (from percent_id_dict, missing treated as -inf).
#          - If drop_duplicates == 'random': pick one row at random per group (stable given random_state).
#     4) Enforce per-species coverage: drop species that appear in fewer than
#        ceil(n_unique_gene_pos * species_threshold) unique gene positions.
#     5) Return the original IDs (column '__ids') as a list.
#     """

#     # ---- Parse ----
#     parsed = _parse_ids(ortholog_ids)  # expects columns: __ensembl_id, __species, __gpos, __ids
    
#     if parsed.empty:
#         return []

#     df = parsed.copy()
#     n_members = df["__gpos"].nunique()

#     # ---- (1) Allowed ortholog proteins for this type ----
#     prot_map = ensembl_homol_reference.get("ensembl_protein_orthologs", {})
#     if ortholog_type == "ortholog_one2one_one2many":
#         allowed_protein_ids = set(prot_map.get("ortholog_one2one", []) + prot_map.get("ortholog_one2many", []))
#     else:
#         allowed_protein_ids = set(prot_map.get(ortholog_type, []))
#     if not allowed_protein_ids: # Nothing allowed → early exit
#         return []

#     # ---- (2) Allowed species by proteome counts + always HUMAN ----
#     spp_counts_by_type = ensembl_homol_reference.get("spp_prefix_counts", {})
#     if ortholog_type == "ortholog_one2one_one2many":
#         spp_counts_all = spp_counts_by_type.get("ortholog_one2one", {}) | spp_counts_by_type.get("ortholog_one2many", {})
#     else:
#         spp_counts_all = spp_counts_by_type.get(ortholog_type, {}) or {}
#     spp_counts_filtered = {
#         spp: int(cnt) for spp, cnt in spp_counts_all.items()
#         if pd.notna(cnt) and int(cnt) > spp_count_proteome_threshold
#     }
#     allowed_spp_prefixes = set(spp_counts_filtered) | {"HUMAN"}

#     # Apply filters 1 & 2
#     df = df[(df["__ensembl_id"].isin(allowed_protein_ids) & df["__species"].isin(allowed_spp_prefixes)) | (df["__species"]=='HUMAN')].copy() 
#     if df.empty:
#         return []    
    
#     # (3) Exclude duplicates within (__species, __gpos)
#     duplicates = df[df.duplicated(subset=['__species','__gpos'],keep=False)]
    
#     if duplicates.any().any():
#         if drop_duplicates=='percent_id':
#             percent_id_dict = ensembl_homol_reference['percent_ids']
            
#             duplicates['percent_id'] = duplicates['__ensembl_id'].apply(lambda x: percent_id_dict.get(x, None))
#             #add to filtered ids
#             df = pd.concat([duplicates,df])
#             #drop duplicates with lower percent id
#             df = df.sort_values(by=['__species','__gpos','percent_id'], ascending=[True,True,False]).drop_duplicates(subset=['__species','__gpos'], keep="first")
#         elif drop_duplicates=='random':
#             df = df.drop_duplicates(subset=['__species','__gpos'], keep="first")
#         else:
#             raise ValueError("drop_duplicates must be 'percent_id' or 'random'")

#     if df.empty:
#         return []

#     if ref:
#         # ---- (4) Enforce species coverage threshold ----
#         # Exclude species prefixes with counts < coverage threshold, relative to total n members (idrs in family)
#         # require each species to cover at least ceil(n_members * species_threshold) distinct gene positions
#         min_required = math.ceil(n_members * float(species_threshold))

#         # compute per-species unique __gpos counts, align back
#         spp_unique_counts = df.groupby("__species")["__gpos"].nunique()       
#         mask = spp_unique_counts >=min_required
#         df = df[df['__species'].isin(set(spp_unique_counts[mask].index))]
#         if df.empty:
#             return []
        
    
#     #TODO ADD FILTER FOR N ORTHOLOGS per fam (at least 50?)
#     if len(set(df['__species'])) < n_orthologs:
#         return []

#     print(len(df['__species'].unique()))
    
#     # ---- (5) Return the original IDs ----
#     return df["__id"].tolist()


def filter_orthologs(
    ortholog_ids: Iterable[str],
    ensembl_homol_reference: Dict = read_pickle(CONFIG_DATA['ensembl_homol_ref']),
    ortholog_type: Literal["ortholog_one2one", "ortholog_one2many", "ortholog_one2one_one2many",] = "ortholog_one2one",
    species_threshold: float = 0.5,
    drop_duplicates: Literal["random", "percent_id"] = "percent_id",
    #spp_count_proteome_threshold: int = 1000,
    n_orthologs:int=10,
) -> List[str]:
    """
    Filter a batch of ortholog region IDs given homology references and a per-protein %identity map.
    
    Args:
        ortholog_ids: Iterable of ortholog region ID strings.
        ensembl_homol_reference: Dictionary containing homology reference data.
        ortholog_type: Type of ortholog relationship to filter by.
        species_threshold: Minimum fraction of orthologous segments in a family a species must cover.
        drop_duplicates: Method to use for deduplication ('random' or 'percent_id').
        ##spp_count_proteome_threshold: Minimum proteome count threshold for species inclusion.
        n_orthologs: Minimum number of unique ortholog species required after filtering.

    Steps
    -----
    1) Keep only rows whose Ensembl protein is in allowed orthologs of `ortholog_type`.
    2) Keep only species whose total proteome count for this `ortholog_type` exceeds `spp_count_proteome_threshold`,
       plus always keep HUMAN.
    3) Deduplicate by (__species, __gpos):
         - If drop_duplicates == 'percent_id': keep the row with the highest percent_id
           (from percent_id_dict, missing treated as -inf).
         - If drop_duplicates == 'random': pick one row at random per group (stable given random_state).
    4) Enforce per-species coverage: drop species that appear in fewer than
       ceil(n_unique_gene_pos * species_threshold) unique gene positions.
    5) Return the original IDs (column '__ids') as a list.
    """

    ensembl_homol_reference = read_pickle(CONFIG_DATA['ensembl_homol_ref'])
       
    # ---- Parse ortholog IDs ----
    parsed = _parse_ids(ortholog_ids)  # expects columns: __ensembl_id, __species, __gpos, __ids
    
    if parsed.empty:
        return []

    df = parsed.copy()
    n_members = df["__gpos"].nunique()

    # ---- (1) Allowed ortholog proteins for this ortholog type ----
    prot_map = ensembl_homol_reference.get("ensembl_protein_orthologs", {})
    if ortholog_type == "ortholog_one2one_one2many":
        allowed_protein_ids = set(prot_map.get("ortholog_one2one", []) + prot_map.get("ortholog_one2many", []))
    else:
        allowed_protein_ids = set(prot_map.get(ortholog_type, []))
    if not allowed_protein_ids: # Nothing allowed → early exit
        return []

    # ---- (2) Allowed species by proteome counts + always HUMAN ----
    spp_counts_by_type = ensembl_homol_reference.get("spp_prefix_counts", {})
    if ortholog_type == "ortholog_one2one_one2many":
        spp_counts_all = spp_counts_by_type.get("ortholog_one2one", {}) | spp_counts_by_type.get("ortholog_one2many", {})
    else:
        spp_counts_all = spp_counts_by_type.get(ortholog_type, {}) or {}
    spp_counts_filtered = {
        spp: int(cnt) for spp, cnt in spp_counts_all.items()
        if pd.notna(cnt) #and int(cnt) > spp_count_proteome_threshold
    }
    allowed_spp_prefixes = set(spp_counts_filtered) | {"HUMAN"}

    # Apply filters 1 & 2
    df = df[(df["__ensembl_id"].isin(allowed_protein_ids) & df["__species"].isin(allowed_spp_prefixes)) | (df["__species"]=='HUMAN')].copy() 
    if df.empty:
        return []    
    
    # (3) Exclude duplicates within (__species, __gpos)
    duplicates = df[df.duplicated(subset=['__species','__gpos'],keep=False)]
    
    if duplicates.any().any():
        if drop_duplicates=='percent_id':
            percent_id_dict = ensembl_homol_reference['percent_ids']
            
            duplicates['percent_id'] = duplicates['__ensembl_id'].apply(lambda x: percent_id_dict.get(x, None))
            #add to filtered ids
            df = pd.concat([duplicates,df])
            #drop duplicates with lower percent id
            df = df.sort_values(by=['__species','__gpos','percent_id'], ascending=[True,True,False]).drop_duplicates(subset=['__species','__gpos'], keep="first")
        elif drop_duplicates=='random':
            df = df.drop_duplicates(subset=['__species','__gpos'], keep="first")
        else:
            raise ValueError("drop_duplicates must be 'percent_id' or 'random'")

    if df.empty:
        return []

    if species_threshold:
        # ---- (4) Enforce species coverage threshold ----
        # Exclude species prefixes with counts < coverage threshold, relative to total n members (idrs in family)
        # require each species to cover at least ceil(n_members * species_threshold) distinct gene positions
        min_required = math.ceil(n_members * float(species_threshold))

        # compute per-species unique __gpos counts, align back
        spp_unique_counts = df.groupby("__species")["__gpos"].nunique()       
        mask = spp_unique_counts >=min_required
        df = df[df['__species'].isin(set(spp_unique_counts[mask].index))]
        if df.empty:
            return []
        
    
    #TODO ADD FILTER FOR N ORTHOLOGS per fam (at least 50?)
    if len(set(df['__species'])) < n_orthologs:
        return []

    #print(len(df['__species'].unique()))
    # ---- (5) Return the filtered IDs ----
    return df["__id"].tolist()

from typing import Dict, Iterable, List, Literal, Optional
def filter_dist_by_orthologs(ortho_ids, dist_res:Dict,
                            ortholog_type = "ortholog_one2one", 
                            species_threshold = 0.5,
                            drop_duplicates = "percent_id",
                            #spp_count_proteome_threshold = 1,
                            n_orthologs=10):
    #dist_res = Dict of distances for a family
    #ortho_ids = list of full ortholog ids for that family (spp_gene_pos)
    
    #Post filtering distances
    
    filtered_ids = filter_orthologs(ortholog_ids=ortho_ids,
                                        ortholog_type = ortholog_type, 
                                        species_threshold = species_threshold,
                                        drop_duplicates = drop_duplicates ,
                                        #spp_count_proteome_threshold =  spp_count_proteome_threshold,
                                        #ref=False
                                        )
        
    parsed_df = _parse_ids(filtered_ids)
    
    dist_res = {key:dict(v) for key, v in dist_res.items()}
    
    filtered_spp = set(parsed_df['__species'])
    filtered_gpos = set(parsed_df['__gpos'])
    
    filtered_dist_res = {pair:{s:v for s,v in orthologs.items() if s in filtered_spp} for pair,orthologs in dist_res.items() if all(gpos in filtered_gpos for gpos in pair.split('|'))}
    
    #Check number of unique orthologs found overall
    unique_ortho_spp = {spp for pair,orthologs in dist_res.items() for spp in list(orthologs.keys()) if len(str(spp))>3}

    if len(unique_ortho_spp) < n_orthologs:
        print(f'n orthologs in distances less than {n_orthologs}...Removing...')
        #remove family from list
        return 0
    else:
        #Take mean of distances per ortholog
        filtered_dist_res = {pair: {spp:np.mean(dists) for spp, dists in value.items()} for pair, value in filtered_dist_res.items()}
        return filtered_dist_res



class ComputeNNDivergence():
    def __init__(self,embed_df):
        
        #embed_df = pd.read_csv(human_idr_embed_path,header=None)
        embed_df[0] = embed_df[0].apply(lambda x: f'HUMAN_{x}')
        embed_df[0] = embed_df[0].str.upper()
        
        self.emb_matrix = EmbedDistanceMatrix(embed_df)
        self.id_col = 0
        self.pos = -4
        self.ids =self.emb_matrix.distance_ids
        self.embed_df = embed_df
    
    def _parse_ids(self,ortholog_df):
        """
        Vectorized parse of ID strings into __species and __gpos columns.
        Assumes IDs look like 'SPECIES_..._...'; __gpos is the join of the last |pos| tokens.
        """
        pos = self.pos
        id_col = self.id_col
        ids = ortholog_df.iloc[:, id_col].astype(str)
        parts = ids.str.split('_')                      # Series[list[str]]

        k = abs(pos)
        if k == 0:
            raise ValueError("pos must be non-zero; use negative values to take last k tokens")

        # join the last k tokens (vectorized for list-like via .str.slice)
        gpos = parts.str.slice(-k).str.join('_')
        genes = parts.str.get(-k)

        species_full = parts.str.get(0)                 # first token

        # remove non-letters; if result is empty, fall back to original
        species_clean = species_full.str.replace(r'[^A-Za-z]', '', regex=True)
        species = species_clean.where(species_clean.ne(''), species_full)
        
        df = ortholog_df.copy()
        df['__id'] = ids
        df['__species'] = species
        df['__genes'] = genes
        df['__gpos'] = gpos
        return df#, ids.name # return the original id column name too

    def _to_scalar(x):
        return float(np.mean(x)) if getattr(x, "size", 0) else float(x)
        
    
    def precompute_ortholog_df(self, ortholog_df):
        emb_matrix=self.emb_matrix
        added_ids = np.array(list(ortholog_df[self.id_col]))
        new_dm, updated_ids = emb_matrix.add_to_distance_matrix(ortholog_df)
        
        return added_ids,new_dm,updated_ids
    
    def between_segment_divergence_single_spp(self, genepos_list):
        # distances = defaultdict(dict)
        # df = self._parse_ids(self.embed_df)
        # grp = df.groupby('__gene')['__id'].apply(list).to_dict()
        
        # grp.get(g1)
        # gene_regions = get_gene_name_features(names=self.ids)
        
        distance_matrix = self.emb_matrix.distance_matrix
        ids = self.ids
        select_ids = [id for id in ids if any(g in id for g in genepos_list)]
        
        id_to_idx, _ = _build_index_maps(ids=ids)
        base_sorted_vals = _precompute_base_sorted(distance=distance_matrix)
        
        #Get pairwise distances:
        combinations_ids = combinations(select_ids,2)
        
        nn_distances = [find_NN_pair_fast(pair[0], pair[1], distance_matrix, id_to_idx, base_sorted_vals) 
                   for pair in combinations_ids]
        
        return np.array(nn_distances)
    
    
    def between_ortholog_divergence(self,gene_pos_list,ortholog_df,precompute_orthologs=True,
                                    ortholog_filter=False,
                                    ortholog_type="ortholog_one2one",
                                    species_threshold=0.5,
                                    chunk_size=10):
        
        id_col =self.id_col

        # Ensure the ID column is string-typed and keep the same order as ids_for_matrix
        ortholog_df[id_col] = ortholog_df[id_col].astype(str)
        
        if ortholog_filter:
            filtered_ids = filter_orthologs(ortholog_ids = list(set(ortholog_df[id_col])),
                                            ortholog_type = ortholog_type,
                                            species_threshold = species_threshold,
                                            ref=True)
    
            ortholog_df = ortholog_df[ortholog_df[id_col].isin(filtered_ids)]
            
            if ortholog_df.empty:
                print(f'No orthologs passed filtering criteria:{ortholog_type}, species threshold per family:{species_threshold}. Gene pos list:{gene_pos_list}')
                return {}                             
        
        df= self._parse_ids(ortholog_df)
        
        # Build group map (species, gpos) -> list of IDs
        grp = df.groupby(['__species', '__gpos'])['__id'].apply(list).to_dict()
        
        all_species = sorted({sp for sp, _ in grp.keys() if sp != 'HUMAN'})
        gene_pos_set = set(gene_pos_list)
        distances = defaultdict(dict)                        
        
        if precompute_orthologs:
            added_ids,new_dm,updated_ids = self.precompute_ortholog_df(ortholog_df)
            
        #n = len(gene_pos_set)
        #len_combos = comb(n, 2)
        
        #New addition
        if chunk_size:
            chunks = chunk_list(list(combinations(gene_pos_set,2)),chunk_size)
        else:
            chunks = combinations(gene_pos_set,2)
        
        len_combos =len(chunks)
            
        for i, chunk in enumerate(chunks):
            print(i, chunk)
            for j, (gp1, gp2) in enumerate(chunk):
        
        #for i, (gp1, gp2) in enumerate(combinations(gene_pos_set,2), start=1):
                #print(gp1,gp2)  
                print(f'chunk {i+1}/{len_combos}: {j+1}/{len(chunk)}')
                #print(f'{i}/{len_combos}')
                
                H1 = grp.get(('HUMAN',    gp1), [])
                H2 = grp.get(('HUMAN',    gp2), [])
                S1 = [s for sp in all_species for s in grp.get((sp, gp1),[]) if sp != 'HUMAN']
                S2 = [s for sp in all_species for s in grp.get((sp, gp2),[]) if sp != 'HUMAN']
                
                if not (H1 and H2 and S1 and S2): #both have at least 1 item
                    continue
                
                if not precompute_orthologs:
                    ids_for_matrix =  S1+S2 
                    # Create a small DF with just those IDs to feed your emb_matrix method
                    mask = df['__id'].isin(ids_for_matrix)
                    small_df = ortholog_df.loc[mask].copy()
                
                    if small_df.empty:
                        print(f'No embeddings selected for {gp1,gp2}')
                        continue
        
                    added_ids,new_dm,updated_ids = self.precompute_ortholog_df(small_df)
                
                # ranks for all pairs
                ranks1 = compute_groupwise_NN_distances_FAST(
                    group1_ids=H1,
                    group2_ids=S2,
                    added_ids=added_ids,
                    all_ids=list(updated_ids),
                    distance_matrix=new_dm,
                    return_dict=True
                )
                ranks2 = compute_groupwise_NN_distances_FAST(
                    group1_ids=H2,
                    group2_ids=S1,
                    added_ids=added_ids,
                    all_ids=list(updated_ids),
                    distance_matrix=new_dm,
                    return_dict=True
                )
                
                res1= {ortholog: dist for pair, dist in ranks1.items() for ortholog in all_species if ortholog in pair}
                res2= {ortholog: dist for pair, dist in ranks2.items() for ortholog in all_species if ortholog in pair}

                agg = defaultdict(list)
                
                for ranks in (res1,res2):
                    for spp,dist in ranks.items():
                        agg[spp].append(dist)

                # mean per species COMMENTE OUT 11-11-25
                #res = {k: float(np.mean(v)) for k, v in agg.items()}
                res=agg
                
                name = f"{gp1}|{gp2}"
                distances[name] = res
        
        return distances  
        
            
    def within_ortholog_divergence(self,gene_pos_list,ortholog_df,precompute_orthologs=True, as_dict=False,
                                    ortholog_filter=True,
                                    ortholog_type="ortholog_one2one",
                                    species_threshold=0.5):
        
        id_col=self.id_col
        gene_pos_set = set(gene_pos_list)

        # Ensure the ID column is string-typed and keep the same order as ids_for_matrix
        ortholog_df[id_col] = ortholog_df[id_col].astype(str)
        
        if ortholog_filter:
            print('IDs before filtering:', list(ortholog_df[id_col]))
            filtered_ids = filter_orthologs(ortholog_ids = list(set(ortholog_df[id_col])),
                                            ortholog_type = ortholog_type,
                                            species_threshold = species_threshold,
                                            ref=False)
            
            ortholog_df = ortholog_df[ortholog_df[id_col].isin(filtered_ids)]
            print('IDs AFTER filtering:', list(ortholog_df[id_col]))
            if ortholog_df.empty:
                print(f'No orthologs passed filtering criteria:{ortholog_type}, species threshold per family:{species_threshold}. Gene pos list:{gene_pos_list}')
                return {}
        
        df = self._parse_ids(ortholog_df)
        
        # Build group map (species, gpos) -> list of IDs
        grp = df.groupby(['__species', '__gpos'])['__id'].apply(list).to_dict()
        
        all_species = sorted({sp for sp, _ in grp.keys() if sp != 'HUMAN'})
        spp_dist_dicts_ls = []
        spp_dist_dicts = defaultdict(dict)
        
        
        if precompute_orthologs:
            added_ids,new_dm,updated_ids = self.precompute_ortholog_df(ortholog_df)
        
        for gp in gene_pos_set:
 
            human_id = grp.get(('HUMAN', gp),[])
            spp_ids = [s for sp in all_species for s in grp.get((sp, gp),[]) if sp != 'HUMAN']
            
            if not human_id or not spp_ids: #both have at least 1 item
                continue
            
            ids_for_matrix =  spp_ids 
            mask = df['__id'].isin(ids_for_matrix)
            small_df = ortholog_df.loc[mask].copy()

            if small_df.empty:
                print(f'No selected genes for {ortholog}')
                continue
            
            if not precompute_orthologs:
                added_ids,new_dm,updated_ids =self.precompute_ortholog_df(small_df)
                
            
            # ranks for all pairs
            ranks = compute_groupwise_NN_distances_FAST(
                group1_ids=human_id,
                group2_ids=spp_ids,
                added_ids=added_ids,
                all_ids=list(updated_ids),
                distance_matrix=new_dm,
                return_dict=True
            )
            
            res ={}
            
            for ortholog in all_species:
                for pair, dist in ranks.items():
                    if ortholog in pair:
                        res[ortholog] = dist
            
            if as_dict:
                spp_dist_dicts[gp] = res
            
            else:
                spp_dist_dicts_ls.append(res)
        
        if as_dict:
            return spp_dist_dicts
        
        else:
            distances = combine_dicts(spp_dist_dicts_ls)
            return distances
    
    def random_within_ortholog_divergence(self,sample_size=100,chunk_size=10,
                                          ortho_dir =CONFIG_DATA['idr_ortho_embed_dir'],
                                          return_full_dict=False ,
                                          select_spp=None,
                                          is_filter=False
                                          ):
        #TODO Option to filter by species?
        
        all_ids = self.emb_matrix.ids
        sample_size =int(sample_size)
        select_gene_pos = extract_IDs_from_end(random.sample(all_ids,sample_size), select='gene_pos')['gene_pos']

        rand_within_ortho_dist_ls = []
        
        for i,chunk in enumerate(np.array_split(select_gene_pos,chunk_size)):
            print(f'{i+1}/{chunk_size}')
            
            ortholog_df_chunk = pd.concat([embedding_h5_to_dataframe(path) for path in find_files_by_identifier(root_path =ortho_dir, select_ids=chunk, by='gene_pos')])
            
            if select_spp:
                ortholog_df_chunk[ortholog_df_chunk[0].str.contains('|'.join(select_spp))]
            
            res =self.within_ortholog_divergence(ortholog_df=ortholog_df_chunk , gene_pos_list = chunk,as_dict=return_full_dict,
                                                 ortholog_filter=is_filter)
            rand_within_ortho_dist_ls.append(res)
        
        if return_full_dict:
            rand_within_ortho_dist =merge_dict(rand_within_ortho_dist_ls)
        else:
            rand_within_ortho_dist = combine_dicts(rand_within_ortho_dist_ls)
        
        # out= f'/home/moseslab/denise/Thesis/results/NN_distance/test/random_idrs_within_ortho_full_09_15_size{sample_size}'
        # print(f'Saving to {out}...')
        # save_pickle(data=rand_within_ortho_dist,outpath = out )
        
        return rand_within_ortho_dist
    
    def random_between_ortholog_divergence(self,sample_size,n_samples=50, save_path=None,
                                           ortho_dir =CONFIG_DATA['idr_ortho_embed_dir'], return_full_dict=False, select_spp=None,
                                           chunk_size=None,
                                           is_filter=False):
        '''Sample_size = number of genes in a sample
        n_sample = number of times to resample
        is_filter = whether to filter orthologs based on default criteria'''
        
        
        all_ids = self.emb_matrix.ids
        rand_bw_ortho_dist_ls = []
        sample_size=int(sample_size)
        n_samples=int(sample_size)
        
        
        for n in range(n_samples):

            select_gene_pos = extract_IDs_from_end(random.sample(all_ids,sample_size), select='gene_pos')['gene_pos']
            ortholog_df = pd.concat([embedding_h5_to_dataframe(path) for path in find_files_by_identifier(root_path =ortho_dir, select_ids=select_gene_pos, by='gene_pos')])
            
            if select_spp:
                ortholog_df[ortholog_df[self.id_col].str.contains(spp for spp in select_spp)]
            
            res =self.between_ortholog_divergence(ortholog_df=ortholog_df , gene_pos_list = select_gene_pos,
                                                  ortholog_filter=is_filter)
            
            if res:
                distances =  {k:np.mean(v) for k, v in combine_dicts(res.values()).items()} #{k:np.mean(v.values()) for k,v in res.items()}
                rand_bw_ortho_dist_ls.append(distances)
                
        rand_between_ortho_dist = combine_dicts(rand_bw_ortho_dist_ls)
        
        if save_path:
            outpath= f'{save_path}_size{sample_size}'
            print(f'Saving to {outpath}...')
            save_pickle(data=rand_between_ortho_dist,outpath = outpath )
        
        return rand_between_ortho_dist
    

#Functions to compute NN divergence from protein families dict

# def differences(dict1,dict2):
#     fam_dist_dict = dict1
#     unrelated_dist_dict =dict2
    
#     diff: Dict[str, float] = {}
    
#     common_keys = fam_dist_dict.keys() & unrelated_dist_dict.keys()
    
#     for ortho in common_keys:
#         fam_val = fam_dist_dict.get(ortho, ())
#         bg_val  = unrelated_dist_dict.get(ortho, ())
        
#         if fam_val.size == 0 or bg_val.size == 0:
#             continue
        
#         diff[ortho] = fam_val - bg_val
#     return diff


def differences(fam_dist: Dict[str, np.ndarray], #TODO is is np array or float?
                bg_dist: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute element-wise differences between matching entries in two dictionaries.
    For each shared key (ortholog pair), returns:
        fam_dist[key] -  bg_dist[key]
    Keys with missing or empty values are skipped.

    Parameters
    ----------
    fam_dist : dict
        Dictionary of primary distance/divergence values.
    bg_dist : dict
        Dictionary of comparison (background) values.

    Returns
    -------
    Dictionary of differences for shared keys (ortholog pairs).
    """
    
    diff: Dict[str, np.ndarray] = {}

    common_keys = fam_dist.keys() & bg_dist.keys()

    for ortho in common_keys:
        fam_val = fam_dist.get(ortho, ())
        bg_val  = bg_dist.get(ortho, ())

        if fam_val.size == 0 or bg_val.size == 0:
            continue

        diff[ortho] = fam_val - bg_val

    return diff


def _load_distance_dict(dist_path: str) -> Dict[str, Dict[str, List[float]]]:
    """Load a distance dictionary from a pickle file.
    Args:
        dist_path: Path to the pickle file containing the distance dictionary."""
    from utils.helpers import read_pickle, combine_dicts
    
    if isinstance(dist_path,dict):
        dist_dict = combine_dicts(list(dist_path.values()))
        
    elif os.path.isfile(dist_path) and dist_path.endswith('.pkl'):
        dist_dict = read_pickle(dist_path)
        dist_dict = combine_dicts(list(dist_dict.values()))
    elif os.path.isdir(dist_path):
        dist_paths = glob.glob(f'{dist_path}/*')
        dist_dict = {path.split('/')[-1].split('_')[0]: combine_dicts(list(read_pickle(path).values())) for path in dist_paths}
        
    else:
        raise ValueError("dist_path must be a path to a pickle file or a directory of pickle files.")
    return dist_dict


        
    
    
# def divergence_stat_summary(
#     fam_dist_dict: PathOrDict,
#     unrelated_dist_dict: PathOrDict,
#     data_transform: Transform = "mean",
#     save_path: Optional[str] = None,
# ) -> pd.DataFrame:
#     """
#     Compute divergence summary stats per family.

#     Args:
#         fam_dist_dict: Either
#           - dict of {fam_id: fam_dist_dict}, OR
#           - path to directory of per-family pickles, OR
#           - path to a single family pickle.
#         unrelated_dist_dict: Either
#           - dict of unrelated distances, OR
#           - path to unrelated distances pickle.
#         data_transform: Summary transform applied before computing differences.
#         save_path: Optional CSV output path.

#     Returns:
#         DataFrame with one row per family.
#     """
#     unrelated = _load_unrelated(unrelated_dist_dict)
#     all_fams = _load_families(fam_dist_dict)

#     # normalize unrelated
#     unrelated = combine_dicts(unrelated.values())
#     unrelated_summ = transform_data(unrelated, type_=data_transform)

#     rows = []
#     for fam_id, fam in all_fams.items():
#         if fam is None:
#             continue

#         # normalize family (skip if empty)
#         try:
#             fam_flat = combine_dicts(list(fam.values()))
#         except Exception:
#             # if fam isn't nested dict-like, you may want to just use it directly:
#             # fam_flat = fam
#             continue

#         fam_summ = transform_data(fam_flat, type_=data_transform)
#         diff = differences(fam_summ, unrelated_summ)

#         row = {
#             "Group_id": fam_id,
#             f"{data_transform} Bw paralog divergence": fam_summ,
#             "Bw paralog divergence dict": fam_flat,
#             f"{data_transform} scp ortholog divergence": unrelated_summ,
#             f"{data_transform} Divergence_per_ortholog": diff,
#         }

#         if diff:
#             vals = np.asarray(list(diff.values()), dtype=float)
#             row[f"Divergence_{data_transform}_mean"] = float(vals.mean())
#             t_stat, p_val = stats.ttest_1samp(vals, popmean=0)
#             row["1samp_ttest_stat"] = float(t_stat)
#             row["1samp_ttest_pvalue"] = float(p_val)

#         rows.append(row)

#     df = pd.DataFrame(rows)
#     if save_path:
#         df.to_csv(save_path, index=False)
#     return df



# def divergence_stat_summary(fam_dist_dict,
#                             unrelated_dist_dict,
#                             data_transform:Literal['log','log_mean','geo_mean','mean', 'median','log_median','geo_median'] = 'mean',
#                             save_path=None):
#     import os
#     import glob
#     from typing import Dict, Iterable, Literal, Mapping
#     from scipy import stats
#     from utils.helpers import combine_dicts, read_pickle,transform_data
    
#     if isinstance(unrelated_dist_dict,str) and os.path.isfile(unrelated_dist_dict) and unrelated_dist_dict.endswith('.pkl'):
#         unrelated_dist_dict = read_pickle(unrelated_dist_dict)
#     elif isinstance(unrelated_dist_dict,dict) and isinstance(fam_dist_dict,dict):
#         all_fam_dist_dict = fam_dist_dict
#     else:
#         raise ValueError("unrelated_dist_dict must be a path to a pickle file or a dictionary")
    
#     if isinstance(fam_dist_dict,str):
#         if os.path.isdir(fam_dist_dict):
#             fam_dist_paths = glob.glob(f'{fam_dist_dict}/*')
#             all_fam_dist_dict = {path.split('/')[-1].split('_')[0]: read_pickle(path) for path in fam_dist_paths}
#         elif isinstance(fam_dist_dict,str) and os.path.isfile(fam_dist_dict) and fam_dist_dict.endswith('.pkl'):
#             all_fam_dist_dict = {fam_dist_dict.split('/')[-1].split('_')[0]: read_pickle(fam_dist_dict)}
#         else:
#             raise ValueError("fam_dist_dict must be a path to a directory of pickle files or a path to a pickle file.")
    
#     unrelated_dist_dict = combine_dicts(unrelated_dist_dict.values()) # {k:[v for sublist in value for v in sublist] for k,value in unrelated_dist_dict.items()} #Is this necessary?
#     unrelated_summ_data = transform_data(unrelated_dist_dict,type_=data_transform)
    
#     summary = []
    
#     for fam_id, fam_dist_dict in all_fam_dist_dict.items():
        
#         fam_dist_dict = combine_dicts(list(fam_dist_dict.values()))
        
#         row = {'Group_id': fam_id,}
        
#         fam_summ_data = transform_data(fam_dist_dict,type_=data_transform)
#         diff_dict = differences(fam_summ_data,unrelated_summ_data)
        
#         row[f'{data_transform} Bw paralog divergence'] = fam_summ_data
#         row['Bw paralog divergence dict'] = fam_dist_dict
#         row[f'{data_transform} scp ortholog divergence'] = unrelated_summ_data
#         row[f'{data_transform} Divergence_per_ortholog'] = diff_dict

#         if len(diff_dict) > 0:
#             row[f'Divergence_{data_transform}_mean'] = np.mean(list(diff_dict.values()))
#             t_stat, p_val = stats.ttest_1samp(list(diff_dict.values()), popmean=0)
#             #t_stat2, p_val2 = stats.ttest_ind(list(fam_dist_dict.values()), group2)
#             row['1samp_ttest_stat'] = t_stat
#             row['1samp_ttest_pvalue'] = p_val
            
#         if save_path:
#             # Convert to DataFrame and save to CSV
#             family_divergence_df = pd.DataFrame([row])
#             family_divergence_df.to_csv(save_path, index=False, mode='a', header=not os.path.exists(save_path))
#             family_divergence_df = None
        
#         summary.append(row)
    
#     return pd.DataFrame(summary)


##HELPERS

def _load_embeddings(embed_path: Optional[str] = None) -> pd.DataFrame: #segment: SegmentType, 
    """Load the embedding DataFrame from the specified path or default path based on segment."""
    # if embed_path==None:
    #     if segment == 'idr':
    #         embed_path = human_idr_embed_path
    #     elif segment =='domain': #TODO
    #         embed_path = human_pfam_embed_path
    embed_df = pd.read_csv(embed_path,header=None)
    return embed_df

def _load_ortholog_embeddings(ortho_path: str, select_gene_ids: Iterable[str]) -> pd.DataFrame:
    """Load ortholog embeddings from the specified directory for the given uniprot gene IDs.
    Args:
        ortho_dir: Directory containing ortholog embedding files.
        select_gene_ids: Iterable of uniprot gene IDs to select.
    Returns:
        DataFrame containing the concatenated ortholog embeddings."""
        
    import os
    
    if not os.path.isdir(ortho_path):
        ortholog_df = embedding_h5_to_dataframe(ortho_path)
    elif os.path.isdir(ortho_path):
        ortho_dir = ortho_path
        #ortholog_df = pd.concat([embedding_h5_to_dataframe(path) for path in find_files_by_identifier(root_path=ortho_dir, select_ids=select_gene_pos, by='gene_pos')])
        files = find_files_by_prefix(prefixes=select_gene_ids, input_dir =  ortho_dir)
        ortholog_df =  pd.concat([embedding_h5_to_dataframe(file_path) for file_path in files])
        
    return ortholog_df

def _load_protein_families(protein_fam_path: str = CONFIG_DATA['protein_fam_path']) -> Dict[str, List[str]]: #TODO: save as final file instead of preprocessing every time
    """Load protein families from a specified path.
    Args:
        protein_fam_path: Path to the protein families file."""
    #from config import gene_group_df, protein_families
    #from itertools import islice
    gene_group_df = pd.read_csv(protein_fam_path)
    gene_group_df.set_index('Group_id',inplace=True)
    gene_group_df.loc[gene_group_df['Category']=='Family','uniprot_ids'] = gene_group_df.loc[gene_group_df['Category']=='Family','uniprot_ids'].apply(lambda x: eval(x))
    protein_families = gene_group_df.loc[gene_group_df['Category']=='Family','uniprot_ids'].to_dict()
    
    select_prot_fam_ids  = set(gene_group_df[(gene_group_df['n_idrs']>3) & (gene_group_df['n_idrs']<=100)].index.sort_values())
    select_prot_fams = {k:v for k,v in protein_families.items() if k in select_prot_fam_ids}

    return select_prot_fams

#Get gene positions available for gene ID?


def compute_ortholog_divergence(engine: ComputeNNDivergence,
                                gene_pos: Iterable[str],
                                dist_type: Literal['between', 'within'],
                                ortholog_df: pd.DataFrame,
                                precompute_orthologs: bool =True,
                                ortholog_filter: bool = False):
    """Compute ortholog divergence using the specified distance type (within or between orthologs) and segment."""
    
    if dist_type == "between":
        return engine.between_ortholog_divergence(
            gene_pos_list=gene_pos,
            ortholog_df=ortholog_df,
            precompute_orthologs=precompute_orthologs,
            ortholog_filter=ortholog_filter,
        )
    elif dist_type == "within":
        return engine.within_ortholog_divergence(
            gene_pos_list=gene_pos,
            ortholog_df=ortholog_df,
            precompute_orthologs=precompute_orthologs,
            as_dict=False #TODO would I want to change this?
        )
    else:
        raise ValueError(f"Invalid dist_type: {dist_type}")

## PIPELINES TO COMPUTE DIVERGENCE (paralog divergence or ortholog divergence)
## Output = pickled dictionary of pairwise neighbour distances 

def run_gene_divergence_pipeline(
    gene_ids,
    out_path,
    *,
    dist_type="between",
    #segment="idr",
    human_embed_path=None,
    ortholog_path=None
):
    # 1) load
    embed_df = _load_embeddings(human_embed_path)
    engine = ComputeNNDivergence(embed_df)
    ortho_df = _load_ortholog_embeddings(ortho_path = ortholog_path, select_gene_ids= gene_ids)
    #Overlapping terms for genes ids and ortho df
    parsed = engine._parse_ids(ortho_df)
    gene_pos = set(parsed["__gpos"])
    #gene_pos = set(parsed.loc[parsed["__genes"].isin(gene_ids), "__gpos"])

    # 2) process
    dist = compute_ortholog_divergence(
        engine=engine,
        gene_pos=gene_pos,
        ortholog_df=ortho_df,
        dist_type=dist_type,
    )
    # 3) save
    save_pickle(dist, out_path)


def run_fam_divergence_pipeline(
    fam_ids=None,
    out_dir='/home/moseslab/denise/Paper/res',
    *,
    dist_type="between",
    segment="idr",
    human_embed_path=None,
    ortho_dir=None,
    ortholog_filter=False,
    precompute_orthologs=True
):
    # 1) load
    embed_df = _load_embeddings(human_embed_path)
    engine = ComputeNNDivergence(embed_df)
    protein_families = _load_protein_families() #TODO?
    
    if fam_ids is None:
        fam_ids = list(protein_families.keys())

    for fam_id in fam_ids:
        gene_ids = protein_families.get(fam_id)
        if not gene_ids:
            print(f"No gene IDs found for family {fam_id}, skipping.")
            continue
        ortho_df = _load_ortholog_embeddings(ortho_path=ortho_dir, select_gene_ids=gene_ids)

        parsed = engine._parse_ids(ortho_df)
        gene_pos = set(parsed.loc[parsed["__genes"].isin(gene_ids), "__gpos"])

        # 2) process
        dist = compute_ortholog_divergence(
            engine=engine,
            gene_pos=gene_pos,
            ortholog_df=ortho_df,
            dist_type=dist_type,
            ortholog_filter=ortholog_filter,
            precompute_orthologs=precompute_orthologs
        )

        # 3) save
        out_path = f"{out_dir}/{fam_id}_{dist_type}_{segment}_diverge"
        save_pickle(dist, out_path)

def run_random_divergence_pipeline(
    sample_size,
    out_path,
    *,
    dist_type="between",
    segment="idr",
    embed_path=None,
    ortho_dir=None,
):
    # 1) load
    embed_df = _load_embeddings(segment, embed_path)
    engine = ComputeNNDivergence(embed_df)

    # 2) process
    if dist_type == "between":
        dist = engine.random_between_ortholog_divergence(sample_size=sample_size, ortho_dir=ortho_dir)
    elif dist_type == "within":
        dist = engine.random_within_ortholog_divergence(sample_size=sample_size, ortho_dir=ortho_dir)
    else:
        raise ValueError(f"Invalid dist_type: {dist_type}")

    # 3) save
    save_pickle(dist, out_path)
    return dist


def divergence_stat_summary(fam_dist_dict,
                            unrelated_dist_dict,
                            data_transform:Literal['log','log_mean','geo_mean','mean', 'median','log_median','geo_median'] = 'log_mean',
                            save_path=None):
    from utils.helpers import transform_data
    
    
    unrelated_dist_dict = _load_distance_dict(unrelated_dist_dict)
    unrelated_dist_dict= transform_data(unrelated_dist_dict,type_=data_transform)
    #fam_dist_dict = _load_distance_dict(fam_dist_dict)
    
    fam_dist_dict_ = read_pickle(fam_dist_dict)
    fam_dist_dict = {fam_id:combine_dicts(list(dist_dict.values())) for fam_id, dist_dict in fam_dist_dict_.items()}
    fam_dist_dict_ = None
    
    rows = []
    for fam_id, fam_dists in fam_dist_dict.items():
        if fam_dists is None:
            continue

        fam_summ_data = transform_data(fam_dists, type_=data_transform)
        diff = differences(fam_summ_data, unrelated_dist_dict)
    
        row = {
            "Group_id": fam_id,
            #f"{data_transform} Bw paralog divergence": fam_summ_data,
            #f"{data_transform} scp ortholog divergence": unrelated_dist_dict,
            #f"{data_transform} Divergence_per_ortholog": diff,
            f"{data_transform} FND_per_species": diff,
        }

        if diff:
            vals = np.asarray(list(diff.values()), dtype=float)
            row[f"Divergence_{data_transform}_mean"] = float(vals.mean())
            t_stat, p_val = stats.ttest_1samp(vals, popmean=0)
            row["1samp_ttest_stat"] = float(t_stat)
            row["1samp_ttest_pvalue"] = float(p_val)

        rows.append(row)
        
        if save_path:
            # Convert to DataFrame and save to CSV
            family_divergence_df = pd.DataFrame([row])
            family_divergence_df.to_csv(save_path, index=False, mode='a', header=not os.path.exists(save_path))
            family_divergence_df = None

    return pd.DataFrame(rows)