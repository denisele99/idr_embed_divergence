import pandas as pd
import os
import ast
import sys
from collections import Counter
import numpy as np
import pandas as pd
import os
import csv
# from itertools import combinations
sys.path.append('/home/moseslab/denise/Paper/src')
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
#from .go_enrichment_analysis import load_go_annotations
from utils.helpers import load_config
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



def load_go_annotations(path:str):
    
    #TODO NEED TO PREPROCESS AND SAVE:
    #annot_df = pd.read_table('/home/moseslab/denise/IDR_LM/data/annotations/idr_annotations/uniprotkb_Human_AND_model_organism_9606_2024_08_20.tsv')
    #go_annot = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))
   # 
    #OR
    
    annot_df = pd.read_table(path)
    go_annot = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))
    return go_annot

CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
config_data = load_config(CONFIG_PATH)

go_annotation_dict = load_go_annotations(config_data.get("go_annotations"))

@lru_cache(maxsize=1)
def load_go_hierarchy(obo_path:str = config_data["go_obo"]):
    go_hierarchy = obonet.read_obo(obo_path)
    return go_hierarchy

@lru_cache(maxsize=1)
def load_go_sim_graph(obo_path:str = config_data["go_obo"]):
    G = graph.from_resource(obo_path.split('.obo')[0])
    similarity.precalc_lower_bounds(G)
    return G

graph = load_go_sim_graph()

def go_ID_to_function(go_IDs:list,graph=graph):
    '''
    Converts GO ID to associated function
        input: list of GO IDs
        returns: dict -> {GO ID: function;...} 
    '''
    functions = {ID: graph.nodes[ID]['name'] for ID in go_IDs if ID in graph.nodes}
    
    return functions



#You could make it faster by caching or saving the go terms background?

GO_ANNOTATIONS = load_go_annotations(config_data["go_annotations"])



# def run_go_enrichment():
#     enriched_goterms,enriched_gocounts = go_enrichment(study_ids=NN_ids_short,
#                         go_terms_background=go_background,
#                         gene_id_background=list(go_background.keys()),
#                         filter_alpha=alpha)


# def go_enrichment_kNN(self, query_ids, k=50, alpha=0.05):
#     """
#     Perform GO term enrichment analysis for k-NN groups.

#     Args:
#         query_ids (list): List of query IDs for enrichment analysis.
#         go_annotation_dict (dict): Dictionary mapping gene IDs to GO terms.
#         NN (int): Number of nearest neighbors to consider (default=50).
#         alpha (float): Significance level for enrichment (default=0.05).

#     Returns:
#         tuple: (go_enrichment_dict, go_enrichment_counts)
#     """
#     #ids = self.distance_ids
#     #go_background = {id.split("_")[0]: go_annotation_dict.get(id.split("_")[0], []) for id in ids}
#     go_background = self.go_terms_background
#     kNN_dict = self.find_kNN(query_ids, k=k)

#     go_enrichment_dict = {}
#     go_enrichment_counts = {}

#     for query, NN_ids in kNN_dict.items():
        
#         NN_ids_short =  [id.split('_')[0] for id in NN_ids]
#         enriched_goterms,enriched_gocounts = go_enrichment(study_ids=NN_ids_short,
#                         go_terms_background=go_background,
#                         gene_id_background=list(go_background.keys()),
#                         filter_alpha=alpha)
        
#         #go_enrichment_dict[query] = list(enriched_goterms.keys())
#         go_enrichment_dict[query]  = list(enriched_goterms.keys())
#         go_enrichment_counts[query] = enriched_gocounts
#         #go_enrichment_counts[query] = {
#         #    go: f"{NN_count[go]}/{population_counts[go]}"
#         #    for go in enriched_goterms.keys()
#         #}

#     return go_enrichment_dict, go_enrichment_counts

# def run_go_enrichment_dataframe(self, query_ids, k=50, alpha=0.05, type="counts", geneid_dict=None):
#     #TODO add option to save/append to a dataframe csv
#     """
#     Run GO enrichment and return results as a DataFrame.
#     Convert GO enrichment results into a DataFrame.

#     Args:
#         query_ids (list): List of query IDs for enrichment.
#         NN (int): Number of nearest neighbors (default=50).
#         alpha (float): Significance level for enrichment (default=0.05).
#         type (str): Output format ('counts' or 'function').
#         geneid_dict (dict, optional): Mapping of gene IDs for annotations.

#     Returns:
#         pd.DataFrame: DataFrame containing GO enrichment results.
#     """
#     #enrichment dict with counts
#     #go_enrichment_counts is a dict of dicts
    
#     go_enrichment_dict,go_enrichment_counts = self.go_enrichment_kNN(query_ids, k=k, alpha=alpha)
    
#     df_enrichment = pd.DataFrame({'Significantly Enriched GO terms': go_enrichment_dict, 'GO term counts': go_enrichment_counts})

#     if geneid_dict:
#         df_enrichment.index = df_enrichment.index.map(lambda x: geneid_dict.get(x.split("_")[0], x))
    
#     if type == "function":
#         df_enrichment['Significantly Enriched GO terms'] = df_enrichment['Significantly Enriched GO terms'].apply(lambda x: go_ID_to_function([go])[go] for go in x)
    
#     return df_enrichment