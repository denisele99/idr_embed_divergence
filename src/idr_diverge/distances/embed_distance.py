#Modified from /home/moseslab/denise/Thesis/src/analysis/analyze_NN_dist.py

from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances, pairwise
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


# class EmbeddingLookup(object): 
#     ''' Class for managing and querying embedding databases. Modified from Littmann et al. 2021 goPredSim'''

#     def __init__(self, embedding_db):
#         '''Args:
#             embedding_db (pd.DataFrame): A DataFrame where each row represents an entity's embeddings. 
#                                          The first column contains the entity IDs, and the remaining columns 
#                                          contain the embedding vectors.'''
#         self.embedding_db = {}
#         keys = list(embedding_db.iloc[:,0])
#         embeddings = embedding_db.iloc[:,1:].to_numpy()

#         for i,key in enumerate(keys):
#             self.embedding_db[key] = embeddings[i]

#         # prepare data
#         self.ids, self.raw_data = zip(*self.embedding_db.items())
        
#     def run_embedding_lookup_distance(self, querys, metric):

#         """
#         Calculate embedding distance of all querys against the lookup database.

#         Args:
#             queries (dict or np.ndarray): A dictionary of query IDs to embeddings, or a NumPy array of query embeddings.
#             metric (str): The distance metric to use (e.g., 'euclidean', 'cosine').

#         Returns:
#             np.ndarray: A 2D array of distances between each query and each embedding in the database.
#             list: A list of query IDs.
#         """

#         if metric in pairwise.distance_metrics():
#             if isinstance(querys, dict):
#                 query_ids, raw_data_query = zip(*querys.items())
#             else:
#                 raw_data_query = querys
#                 query_ids = range(0, np.shape(querys)[0])
            
#             raw_data_query = np.array(raw_data_query).squeeze()
#             if len(query_ids) == 1:
#                 raw_data_query = raw_data_query.reshape(1, -1)

#             distances = pairwise_distances(raw_data_query, self.raw_data, metric=metric)
#         else:
#             sys.exit("{} is not a correct distance metric\n"
#                      "See <sklearn.metrics.pairwise.distance_metrics()> "
#                      "for all possible distance metrics".format(metric))
            
#         return distances, query_ids
    

# #Maybe delete?

# def search_embed(query_df, lookup_df):
#     emblookup = EmbeddingLookup(lookup_df)
#     query_embedding_db = EmbeddingLookup(query_df).embedding_db
#     distance, ids = emblookup.run_embedding_lookup_distance(querys=query_embedding_db,metric='cosine')
#     return distance,ids

#TODO: This is the same as EmbeddingLookup? Merge, replace Embeddinglookup in GO enrichment script

class EmbedDistanceMatrix:
    '''Modified from Littmann et al. 2021 goPredSim'''
    def __init__(self, embedding_db: pd.DataFrame):
        """
        Initialize the embedding distance matrix object with a lookup database.

        Args:
            embedding_db (pd.DataFrame): A DataFrame where the first column contains IDs 
                                         and the remaining columns contain embedding vectors.
        """
        # Validate input DataFrame
        if embedding_db.shape[1] < 2:
            raise ValueError("embedding_db must have at least one feature column.")
        if embedding_db.iloc[:, 0].duplicated().any():
            raise ValueError("Duplicate IDs found in the embedding database.") #Can comment out
            embedding_db=embedding_db.drop_duplicates(subset=embedding_db.columns[0])
            
        
        # Store embeddings in a dictionary for easy lookup
        self.embedding_db = dict(zip(embedding_db.iloc[:, 0], embedding_db.iloc[:, 1:].to_numpy()))
        
        # Prepare data for distance matrix computation
        self.ids, self.raw_data = zip(*self.embedding_db.items())
        self.raw_data = np.array(self.raw_data)  # Convert to NumPy array
        
        # Compute the initial distance matrix using all embeddings TODO is this necessary?Delete?
        self.distance_matrix, self.distance_ids = self.run_embedding_lookup_distance(
            queries=self.embedding_db, metric='cosine'
        )

    def run_embedding_lookup_distance(self, queries, metric='cosine'):
        """
        Calculate pairwise distances between query embeddings and the lookup database.

        Args:
            queries (dict or np.ndarray): A dictionary of query IDs to embeddings, 
                                          or a NumPy array of query embeddings.
            metric (str): The distance metric to use (e.g., 'cosine', 'euclidean').

        Returns:
            tuple:
                - distances (np.ndarray): A 2D array of distances between queries and the database.
                - query_ids (list): A list of query IDs.
        """
        # Validate distance metric
        if metric not in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.keys():
            raise ValueError(f"{metric} is not a valid distance metric. Use valid metrics from sklearn.")
        
        # Handle queries as dictionary or NumPy array
        if isinstance(queries, dict):
            query_ids, raw_query_data = zip(*queries.items())
        else:
            raw_query_data = queries
            query_ids = range(len(queries))
        
        raw_query_data = np.array(raw_query_data).squeeze()

        # Ensure 2D array for single query
        if len(raw_query_data.shape) == 1:
            raw_query_data = raw_query_data.reshape(1, -1)

        # Compute pairwise distances
        distances = pairwise_distances(raw_query_data, self.raw_data, metric=metric)
        return distances, query_ids

    def add_to_distance_matrix(self, query_embeddings: pd.DataFrame):
        """
        Extend the existing distance matrix by adding query embeddings.

        Args:
            query_embeddings (pd.DataFrame): A DataFrame where the first column contains query IDs 
                                             and the remaining columns are embedding vectors.

        Returns:
            tuple:
                - final_dist_matrix (np.ndarray): Updated distance matrix.
                - updated_ids (list): List of all IDs (existing and new).
        """
        # Validate input DataFrame
        if query_embeddings.shape[1] < 2:
            raise ValueError("query_embeddings must have at least one feature column.")
        if query_embeddings.iloc[:, 0].duplicated().any():
            #raise ValueError("Duplicate IDs found in query_embeddings.") #Can comment out
            print("Duplicate IDs found in query_embeddings.")
            print(list(query_embeddings[query_embeddings.duplicated(subset=query_embeddings.columns[0])][0]))
            query_embeddings=query_embeddings.drop_duplicates(subset=query_embeddings.columns[0])

        # Extract query IDs and vectors
        query_ids = query_embeddings.iloc[:, 0]
        query_vectors = query_embeddings.iloc[:, 1:].to_numpy()
        
        # Number of query embeddings
        n_query = len(query_vectors)
        
        # Calculate distances between query embeddings and existing embeddings
        dist_to_base = pairwise_distances(query_vectors, np.concatenate((self.raw_data,query_vectors),axis=0), metric='cosine')
        
        # Update the distance matrix
        # 1. Add columns for distances from the base embeddings to the queries
        extended_columns = np.concatenate((self.distance_matrix, dist_to_base.T[:-n_query]), axis=1)
        
        # 2. Add rows for distances from the queries to all embeddings (base + query)
        final_dist_matrix = np.concatenate((extended_columns, dist_to_base), axis=0)
        
        # Update IDs
        updated_ids = list(self.ids) + list(query_ids)
        return final_dist_matrix, updated_ids


#CALCULATE NEIGHBOUR DISTANCES

def _build_index_maps(ids:list, added_ids:np.ndarray|list|None = None):
    id_to_idx = {id_: k for k, id_ in enumerate(ids)}
    
    if added_ids is not None:
        is_added = np.zeros(len(ids), dtype=bool)
        if isinstance(added_ids, (list, tuple, np.ndarray, set)):
            for a in added_ids:
                if a in id_to_idx:
                    is_added[id_to_idx[a]] = True
    else:
        is_added = None
    return id_to_idx, is_added #is_added is a mask
    #else:
    #    return id_to_idx

def _precompute_base_sorted(distance:np.ndarray, is_added:np.ndarray|None =None):
    """
    For each row i, precompute the sorted array of distances over the base allowed set:
      allowed_base(i) = { all non-added j } ∪ { i if i is added }
    Returns:
      base_sorted_vals: list of 1D numpy arrays (one per row)
    """
    n = distance.shape[0]
    base_sorted_vals = [None] * n
    
    if is_added is not None:
        non_added_cols = ~is_added
        for i in range(n):
            # Base allowed = non-added OR (self if self is added)
            base_mask = non_added_cols.copy()
            if is_added[i]:
                base_mask[i] = True  # include self (distance[i,i], typically 0)
            vals = distance[i, base_mask]
            # Replace NaNs with +inf so they sort to the end and never “win”
            if np.isnan(vals).any():
                vals = np.where(np.isnan(vals), np.inf, vals)
            base_sorted_vals[i] = np.sort(vals, kind='mergesort')  # stable, matches "first among ties"
    else:
        for i in range(n):
            vals = distance[i]
            base_sorted_vals[i] = np.sort(vals, kind='mergesort') 
            
    return base_sorted_vals

def _rank_in_row(i, j, distance, base_sorted_vals):
    """
    Rank position (0-based) of column j in row i under the allowed set:
      allowed = (non-added) ∪ {i if i is added} ∪ {j if j is added}
    We compute it by inserting d(i,j) into the pre-sorted base array.
    """
    d = distance[i, j]
    if np.isnan(d):
        return np.inf  # masked/out-of-scope → treat as infinitely far
    # insertion point gives first position among ties (like argsort + np.where(...)[0][0])
    return int(np.searchsorted(base_sorted_vals[i], d, side='left'))

def find_NN_pair_fast(id1, id2, distance, id_to_idx, base_sorted_vals):
    """
    Same as find_NN_pair, but O(log n) per query after precompute (vs O(n log n)).
    Returns the average of the two directed ranks.
    """
    i1 = id_to_idx[id1]
    i2 = id_to_idx[id2]
    r12 = _rank_in_row(i1, i2, distance, base_sorted_vals)
    r21 = _rank_in_row(i2, i1, distance, base_sorted_vals)
    return (r12 + r21) / 2.0

def compute_groupwise_NN_distances_FAST(group1_ids, group2_ids, added_ids, all_ids, distance_matrix, return_dict=False):
    """
    Compute NN distances between two groups of IDs (fast version)
    
    """
    id_to_idx, is_added = _build_index_maps(all_ids, added_ids)
    base_sorted_vals = _precompute_base_sorted(distance_matrix, is_added)

    if return_dict:
        NN_distances= {}
    else:
        NN_distances=[]

    for a, b in product(group1_ids, group2_ids):
        # If either ID is unknown, skip
        if a not in id_to_idx or b not in id_to_idx:
            continue
        
        dist = find_NN_pair_fast(a, b, distance_matrix, id_to_idx, base_sorted_vals)
        if return_dict:
            NN_distances[f'{a}|{b}']=dist
        else:
            NN_distances.append(dist)
    
    if return_dict:
        return NN_distances

    else:
        return np.asarray(NN_distances, dtype=float)