'''Originally modified from Valr, updated 08/12/24'''
#Last updated 07/28/25

import numpy as np
import sys
import pandas as pd
import csv
from collections import defaultdict, Counter
import os
import obonet
from itertools import combinations
from scipy import stats

from goatools.obo_parser import GODag
from goatools.base import get_godag

from pygosemsim import download, graph
from pygosemsim import similarity



from functools import lru_cache
from utils.helpers import load_config


CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
config_data = load_config(CONFIG_PATH)

@lru_cache(maxsize=1)
def load_go_hierarchy(obo_path:str = config_data["go_obo"]):
    go_hierarchy = obonet.read_obo(obo_path)
    return go_hierarchy

@lru_cache(maxsize=1)
def load_go_sim_graph(obo_path:str = config_data["go_obo"]):
    G = graph.from_resource(obo_path.split('.obo')[0])
    similarity.precalc_lower_bounds(G)
    return G

def load_go_annotations(path:str = config_data["go_annotations"]):
    
    #TODO NEED TO PREPROCESS AND SAVE:
    #annot_df = pd.read_table('/home/moseslab/denise/IDR_LM/data/annotations/idr_annotations/uniprotkb_Human_AND_model_organism_9606_2024_08_20.tsv')
    #go_annot = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))
   # 
    #OR
    
    annot_df = pd.read_table(path)
    go_annot = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))
    return go_annot


def read_go_annotations(go_file, column_gene="Entry", column_goIDs="Gene Ontology IDs") -> dict:
    """
    Reads a GO annotation file and returns a dictionary mapping gene names to lists of GO terms.

    Args:
        go_file (str or pd.DataFrame): Path to a CSV file or a DataFrame with GO annotations.
        column_gene (str): Column name containing gene identifiers (default: "Entry").
        column_go (str): Column name containing GO terms (default: "Gene Ontology IDs").

    Returns:
        dict: {gene_name: [GO_ID1, GO_ID2, ...]}
    """
    # Load CSV if a file path is provided
    if isinstance(go_file, str):
        if os.path.exists(go_file):
            if go_file.endswith('.csv'):
                go_file = pd.read_csv(go_file)
            elif go_file.endswith('.tsv'):
                go_file = pd.read_table(go_file)
            else:
                raise TypeError('File must be .csv or .tsv')
        else:
            raise FileExistsError()
    elif not isinstance(go_file, pd.DataFrame):
        raise TypeError('File must be .csv or .tsv path, or a pd.Dataframe')

    # Drop rows with missing gene names or GO annotations
    go_file = go_file.dropna(subset=[column_gene, column_goIDs])

    # Initialize output dictionary
    go_dict = {}

    for _, row in go_file.iterrows():
        gene = str(row[column_gene]).strip()
        go_terms = [term.strip() for term in str(row[column_goIDs]).split(';') if term.strip()]
        
        # Only store if there are valid GO terms
        if gene and go_terms:
            go_dict[gene] = go_terms

    return go_dict


def go_ID_to_function(go_IDs:list,graph=graph):
    '''
    Converts GO ID to associated function
        input: list of GO IDs
        returns: dict -> {GO ID: function;...} 
    '''
    functions = {ID: graph.nodes[ID]['name'] for ID in go_IDs if ID in graph.nodes}
    
    return functions

# class GeneOntology(object):
#     '''From Littmann Gopredsim
#     https://github.com
#     @author: X
#     '''

#     def __init__(self, onto_file):
#         self.all_go = defaultdict(dict)

#         self._parse_go(onto_file)
#         self.mfo = self._get_go_annotations('mfo')
#         self.bpo = self._get_go_annotations('bpo')
#         self.cco = self._get_go_annotations('cco')

#     def _parse_go(self, onto_file):
#         # information to save for a GO term
#         go_id = ''
#         go_name = ''
#         namespace = ''
#         parents = set()
#         alt_ids = set()

#         term = False

#         with open(onto_file) as read_in:
#             for line in read_in:
#                 splitted_line = line.strip().split(':')
#                 if '[Term]' in line:  # new term begins
#                     term = True
#                     if not go_id == '':
#                         if go_id in self.all_go.keys():
#                             print(go_id)
#                         self.all_go[go_id] = {'name': go_name, 'go': namespace, 'parents': parents}
#                         for a in alt_ids:
#                             self.all_go[a] = {'name': go_name, 'go': namespace, 'parents': parents}

#                         # reset annotations
#                         go_id = ''
#                         go_name = ''
#                         namespace = ''
#                         parents = set()
#                         alt_ids = set()
#                 elif term and 'id: GO:' in line and 'alt_id' not in line:
#                     go_id = "GO:{}".format(splitted_line[2].strip())
#                 elif term and 'alt_id: GO' in line:
#                     alt_id = "GO:{}".format(splitted_line[2].strip())
#                     alt_ids.add(alt_id)
#                 elif term and 'name:' in line:
#                     go_name = splitted_line[1].strip()
#                 elif term and 'namespace:' in line:
#                     tmp_nampespace = splitted_line[1].strip()
#                     if tmp_nampespace == 'biological_process':
#                         namespace = 'bpo'
#                     elif tmp_nampespace == 'molecular_function':
#                         namespace = 'mfo'
#                     elif tmp_nampespace == 'cellular_component':
#                         namespace = 'cco'
#                 elif term and 'is_a:' in line:
#                     splitted_term = splitted_line[2].split("!")
#                     go_term = "GO:{}".format(splitted_term[0].strip())
#                     parents.add(go_term)
#                 elif '[Typedef]' in line:
#                     term = False

#         self.all_go[go_id] = {'name': go_name, 'go': namespace, 'parents': parents}

#         # include all parents (also grandparents,...)
#         for go_term in self.all_go.keys():
#             new_parents = self._set_parents(go_term)
#             self.all_go[go_term]['parents'].update(new_parents)

#     def _set_parents(self, term):
#         new_parents = set()

#         parents = self.all_go[term]['parents']
#         for p in parents:
#             tmp_parents = self._set_parents(p)
#             new_parents.update(tmp_parents)
#         new_parents.update(parents)

#         return new_parents

#     def _get_go_annotations(self, onto):
#         ontology = defaultdict(dict)

#         for k in self.all_go.keys():
#             if self.all_go[k]['go'] == onto:
#                 ontology[k] = self.all_go[k]

#         return ontology

#     def get_parent_terms(self, go_term):
#         if go_term in self.all_go.keys():
#             return self.all_go[go_term]['parents']
#         else:
#             return set()

#     def get_all_terms(self, leaf_annotations):
#         all_annotations = defaultdict(set)

#         for k in leaf_annotations.keys():
#             go_terms = leaf_annotations[k]
#             for g in go_terms:
#                 parent_terms = self.get_parent_terms(g)
#                 all_annotations[k].add(g)
#                 all_annotations[k].update(parent_terms)

#         return all_annotations

#     def get_ontology(self, go_term):
#         if go_term in self.all_go.keys():
#             return self.all_go[go_term]['go']
#         else:
#             return ''

#     def get_name(self, go_term):
#         return self.all_go[go_term]['name']



def jaccard_similarity(a,b): #jaccard similarity
    '''Jaccard similarity, where the input is a list of go terms
        Args:
            a(list): list of GO IDs
            b(list): list of GO IDs'''

    if len(a)==0 or len(b)==0:
        return np.nan

    intersection = len(set(a).intersection(b)) #overlapping non-zero positions
    union = len(set(a).union(b))
    #union = (len(a)+len(b)) - intersection #unique non-zero positions in A and B
 
    return intersection / union




# def get_terms_by_go(go_,terms):
#         '''go_: GeneOntology object
#         terms: set of GO terms
        
#         Return: dict of go terms split up by their type'''
#         terms_by_go = {'mfo': set(), 'bpo': set(), 'cco': set()}

#         for t in terms:
#             onto = go_.get_ontology(t)
#             if onto != '':
#                 terms_by_go[onto].add(t)

#         return terms_by_go

# def jaccard_go(gene1,gene2,go_db=go_annotations,go_type='all'):
#     '''measure GO term similarity given gene names and annotation file, filter by cco, bpo, and mfo'''
#     if (isinstance(gene1,list)) or (isinstance(gene1,np.ndarray)):
#         gene1 = gene1[0]
#         gene2 = gene2[0]
#     if ('_' in gene1) or ('_' in gene2):
#         gene1= gene1[:gene1.rfind('_')]
#         gene2= gene2[:gene2.rfind('_')]
#     if go_type == 'all':
#         try:
#             result = jaccard_similarity(go_db[gene1],go_db[gene2])
#         except:
#             result = np.nan
    
#     elif go_type == 'mfo' or go_type == 'bpo' or go_type == 'cco':
#         go_terms1 = get_terms_by_go(go_obo,go_db[gene1])[go_type]
#         go_terms2 = get_terms_by_go(go_obo,go_db[gene2])[go_type]
#         try:
#             result = jaccard_similarity(go_terms1,go_terms2)
#         except:
#             result = np.nan
#     return result


#Analysis/Accuracy of GO enrichment results
def go_enrichment_score(go_enrichment_dict,score,outpath):
    precision_dict = {}
    
    if type=='semsim':
            #G = graph.from_resource("go-basic")
            G = graph.from_resource("/home/moseslab/denise/IDR_LM/data/annotations/go_2024")
            similarity.precalc_lower_bounds(G)
            
    #input is a go dictionary {query: list of enriched GO terms, etc.}
    for query, enriched_goterms in go_enrichment_dict.items():
        query_goterms = go_annotation_dict.get(query.split("_")[0], [])

        if score == "precision":
            precision_dict[query] = precision_fromgo_set(query_goterms, enriched_goterms)
        elif score == "recall":
            precision_dict[query] = recall_fromgo_set(query_goterms, enriched_goterms)
        elif score == "semsim":
            precision_dict[query] = go_sim_sem_score(
                G=G, goset_query=enriched_goterms, goset_ref=query_goterms
            )

        pd.DataFrame.from_dict({query: precision_dict[query]}, orient="index").to_csv(
            outpath, mode="a", index=True, header=False
        )
    return precision_dict


def fold_enrichment(observed_ratio, background_n = 19459, NN=config_data.get("k")):
    observed_ratio = tuple(observed_ratio.split('/'))
    n_term_observed,n_term_background = observed_ratio

    observed = int(n_term_observed)/NN
    expected = int(n_term_background)/background_n
    
    return round(observed/expected,1)


# Helper function to count GO terms in the dataframe
def count_go_terms(df, query):
    go_list = df.loc[df['query'].str.startswith(query) & df['hit_pred'].str.len() > 0, 'hit_pred']
    flattened_list = [item for sublist in go_list for item in sublist]
    return Counter(flattened_list), len(go_list)

# Perform z-test on GO term counts
def z_test(go_dict1, go_dict2, nobs1, nobs2):
    combined_goset = set(go_dict1) | set(go_dict2)
    p_value_ls = {}

    for go_term in combined_goset:
        count1 = go_dict1.get(go_term, 0)
        count2 = go_dict2.get(go_term, 0)
        
        count = np.array([count1, count2])
        nobs = np.array([nobs1, nobs2])
        _, p_value = proportions_ztest(count, nobs)
        
        p_value_ls[go_term] = p_value

    return p_value_ls

def get_precision(TP,FP):
    if (TP+FP)==0:
        return 0
    return TP/(TP+FP)

def precision_fromgo_set(query_go, enriched_go):
    TP = list(set(query_go) & set(enriched_go))
    FP = [go for go in enriched_go if not(go in query_go)]
    
    if (len(TP)<1) & (len(FP)<1):
        return np.nan
    return get_precision(len(TP),len(FP))

def recall_fromgo_set(query_go, enriched_go):
    TP = list(set(query_go) & set(enriched_go))
    FN = [go for go in query_go if not(go in enriched_go)]
    
    if (len(TP)<1) & (len(FN)<1):
        return np.nan
    
    return len(TP)/(len(TP)+len(FN))
    

def go_sim_sem_score(G, goset_query,goset_ref): #retire this function
    
    if (len(goset_query)<1) or (len(goset_ref)<1):
        return np.nan
    
    #Similarity for each feature in goset_query
    sim_dict = {}
    for go in goset_query:
        sim = []
        for go2 in goset_ref:
            try:
                sim.append(similarity.wang(G, go, go2))
            except:
                continue
        if len(sim)>0:
            sim_dict[go] = max(sim)
        #sim_dict[go] = max([similarity.wang(G, go, go2) for go2 in goset_ref])
    
    if len(sim_dict)>0:
        accuracy = sum(sim_dict.values())/len(sim_dict) #TODO - instead of averaging, add up the values? or ratio of correct terms to known terms
        #accuracy = sum(sim_dict.values())/len(goset_ref)
    else:
        accuracy=np.nan
    return accuracy



def wang_similarity(term1, term2,G_graph):
    return similarity.wang(G_graph, term1, term2)
    
def go_bma_similarity(go_terms1, go_terms2):
    
    G = load_go_sim_graph()
    
    if (len(go_terms1)<1) or (len(go_terms2)<1):
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

def get_enrichment_terms(uniprot_ids, go_enrich_dict):
    go_terms_list = [terms for key, terms in go_enrich_dict.items() if len(terms)>0 and any(key.startswith(uid) for uid in uniprot_ids)]
    return go_terms_list


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



# GO term divergence analysis

#Input: go term associations with list of genes, group1_ids, group2_ids
#Output: significantly enriched GO term divergence between group1 and group2

def go_enrichment_divergence(go_term_set1, go_term_set2, alpha_value=0.05):
    #where go term set is a list of lists of go terms
    #Or select go_terms from list of ids
    go_term_set1_flat = [term for sublist in go_term_set1 for term in sublist]
    go_counts_1 = Counter(go_term_set1_flat)
    nobs1 = len(go_term_set1)
    
    go_term_set2_flat = [term for sublist in go_term_set2 for term in sublist]
    go_counts_2 = Counter(go_term_set2_flat)
    nobs2 = len(go_term_set2)
    
    z_res = z_test(go_counts_1, go_counts_2, nobs1, nobs2)
    z_res_filtered = {k: v for k, v in z_res.items() if not np.isnan(v)}
    adjusted_p_values = multipletests(list(z_res.values()), alpha=alpha_value, method='fdr_bh')[1]
    
    z_res_adjusted = dict(zip(z_res_filtered.keys(), adjusted_p_values))
    
    return z_res_adjusted

def go_jaccard_similarity(go_terms1, go_terms2):
    #Similarity of two sets of GO terms
     # Convert lists to sets
    set1 = set(go_terms1)
    set2 = set(go_terms2)
    
    # Calculate the intersection and union of the sets
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    # Calculate Jaccard similarity
    if len(go_terms1) == 0 or len(go_terms2) == 0:
        return 0
    return intersection / union

def go_jaccard_divergence(go_terms1, go_terms2):
    return 1 - go_jaccard_similarity(go_terms1, go_terms2)