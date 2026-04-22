#Copied from /home/moseslab/denise/IDR_LM/src/experiments/final_scripts/calc_go_enrichment.py 06/27/25

from collections import Counter
import numpy as np
import pandas as pd
import os
import csv
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Optional
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from collections import defaultdict, Counter
import obonet
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
from pygosemsim import graph,similarity
from functools import lru_cache
from pathlib import Path


from idr_diverge.distances.embed_distance import EmbedDistanceMatrix
from idr_diverge.utils.helpers import load_config
from idr_diverge.distances.compute_ndist import  _load_embeddings


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "go_config.yaml"
)


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
    annot_df = pd.read_table(path)
    go_annot = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))
    return go_annot


config_data = load_config(DEFAULT_CONFIG_PATH)

GO_ANNOTATIONS = load_go_annotations(config_data.get("go_annotations"))
OBO_PATH = config_data["go_obo"]

@lru_cache(maxsize=1)
def load_go_hierarchy(obo_path:str = OBO_PATH):
    go_hierarchy = obonet.read_obo(obo_path)
    return go_hierarchy

@lru_cache(maxsize=1)
def load_go_sim_graph(obo_path:str = OBO_PATH):
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


def get_enrichment_terms(genes, go_enrich_dict):
    go_terms_list = [terms for key, terms in go_enrich_dict.items() 
                     if len(terms)>0 and any(key.startswith(uid) 
                    for uid in genes)]
    return go_terms_list



def _extract_gene_id(embed_id: str, *, delimiter: str = "_") -> str:
    """Convert an embedding row id like 'GENE_segment_pos1_pos2' -> 'GENE'."""
    return embed_id.split(delimiter, 1)[0]

# def get_go_term_counts(genes, go_annotation_dict):
#     terms= [term for id,terms in go_annotation_dict.items() for term in terms
#         if id in genes]
#     #terms = get_enrichment_terms(genes,go_annotation_dict)
#     term_counts = Counter(terms)
#     return term_counts

def count_go_terms(
    genes: List[str],
    #go_by_gene: Dict[str, List[str]],
    go_annotation_dict: Dict[str, List[str]],
    deduplicate_genes: bool = False,
) -> Counter:
    """
    Count GO terms across a gene set.

    If deduplicate_genes=True, repeated genes count once.
    """
    gene_list = list(genes)
    if deduplicate_genes:
        gene_list = list(set(gene_list))  # stable unique

    terms: List[str] = []
    for g in gene_list:
        #terms.extend(go_by_gene.get(g, ()))
        terms.extend(go_annotation_dict.get(g, ()))
    return Counter(terms)




# def go_enrichment_fisher_1(
#     study_ids: List[str],
#     #background_go_terms_by_id: Dict[str, List[str]],
#     background_ids: List[str],
#     background_term_counts: Dict[str,int] = None,
#     *,
#     alpha: float = 0.01,
#     go_terms_by_id: Dict[str, List[str]] = GO_ANNOTATIONS,
#     term_function:bool=False
# ) -> Tuple[Dict[str, float], Dict[str, str]]:
#     """
#     Run GO enrichment using Fisher's exact test (one-sided by default) and FDR (BH) correction. Optionally filter by adjusted p-value significance(alpha).

#     Parameters
#     ----------
#     study_ids:
#         IDs in the study set (e.g., nearest neighbors).
#     background_go_terms_by_id:
#         Mapping from background gene/ID -> iterable of GO terms.
#         (Used to compute background GO term counts.)
#     background_ids:
#         IDs in the background population (used for totals).
#     alpha:
#         Significance threshold applied after FDR correction. If None, no filtering is applied.
#     go_terms_by_id:
#         Mapping from ID -> iterable of GO terms for the study IDs.

#     Returns
#     -------
#     enriched_terms:
#         Dict mapping GO term -> adjusted p-value (filtered by alpha if provided).
#     term_counts:
#         Dict mapping GO term -> "study_count/background_count" for the returned terms.

#     Notes
#     -----
#     - Tests are performed over GO terms observed in the study set. If you want a more conservative approach, iterate over all background terms instead.
#     """
#     # Study GO term counts
#     # study_terms: List[str] = [term for gene_id in study_ids
#     #                         for term in go_terms_by_id.get(gene_id, ())
#     #                         ]
#     study_term_counts  = count_go_terms(genes=study_ids, go_annotation_dict=go_terms_by_id)
#     study_unique_terms = set(study_term_counts.keys())
#     if not study_unique_terms:
#         raise ValueError(F"Study IDs {study_ids} not found in background_go_terms_by_id")
#         #return {}, {}

#     #study_term_counts = Counter(study_terms)
    

#     # Background GO term counts
#     if background_term_counts == None:                              
#         background_term_counts = count_go_terms(genes = background_ids, go_annotation_dict=go_terms_by_id) #allows duplicates if duplicate genes
#     background_unique_terms = list(background_term_counts.keys())
#     # Totals 
#     study_total = len(study_ids)
#     background_total = len(background_ids)
#     if background_total < study_total:
#         raise ValueError(
#             f"background_ids has fewer IDs ({background_total}) than study_ids ({study_total})."
#         )
#     background_only_total = background_total - study_total

#     # --- Fisher tests ---
#     tests: List[Tuple[str, float]] = []
#     #for term in background_unique_terms:
#     for term in study_unique_terms:
#         a = study_term_counts.get(term, 0)              # in study + has term
#         c_total = background_term_counts.get(term, 0)   # in background + has term
#         c = max(0, c_total - a)                         # in background-only + has term
        
#         # Contingency:
#         #            term     not term
#         # study        a      study_total - a
#         # bg-only      c      background_only_total - c
#         table = [
#             [a, study_total - a],
#             [c, background_only_total - c],
#         ]

#         _, p = fisher_exact(table, alternative="greater")
#         tests.append((term, p))

#     if not tests:
#         return {}, {}

#     #Multiple testing correction (BH/FDR)
#     pvals = np.array([p for _, p in tests], dtype=float)
#     adj_pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
#     term_to_adj_p = {term: float(adj) for (term, _), adj in zip(tests, adj_pvals)}

#     # Optional filtering by adjusted p value
#     if alpha is not None:
#         term_to_adj_p = {t: p for t, p in term_to_adj_p.items() if p < alpha}

#     term_counts = {
#         term: f"{study_term_counts[term]}/{background_term_counts.get(term, 0)}"
#         for term in term_to_adj_p
#     }
    
#     if term_function:
#         terms = list(term_to_adj_p.keys())
#         term_to_function = go_ID_to_function(go_IDs=terms)
#         term_to_adj_p = {term_to_function.get(term,term):val for term, val in term_to_adj_p.items()}
#         term_counts =  {term_to_function.get(term,term):val for term, val in term_counts.items()}

#     return term_to_adj_p, term_counts

# def go_enrichment_fisher(
#     study_genes: List[str],
#     #background_go_terms_by_id: Dict[str, List[str]],
#     background_genes: List[str],
#     background_term_counts: Dict[str,int] = None,
#     *,
#     alpha: float = 0.01,
#     go_terms_by_id: Dict[str, List[str]] = GO_ANNOTATIONS,
#     term_function:bool=False
# ) -> Tuple[Dict[str, float], Dict[str, str]]:
#     """
#     Run GO enrichment using Fisher's exact test (one-sided by default) and FDR (BH) correction. Optionally filter by adjusted p-value significance(alpha).

#     Parameters
#     ----------
#     study_ids:
#         IDs in the study set (e.g., nearest neighbors).
#     background_go_terms_by_id:
#         Mapping from background gene/ID -> iterable of GO terms.
#         (Used to compute background GO term counts.)
#     background_ids:
#         IDs in the background population (used for totals).
#     alpha:
#         Significance threshold applied after FDR correction. If None, no filtering is applied.
#     go_terms_by_id:
#         Mapping from ID -> iterable of GO terms for the study IDs.

#     Returns
#     -------
#     enriched_terms:
#         Dict mapping GO term -> adjusted p-value (filtered by alpha if provided).
#     term_counts:
#         Dict mapping GO term -> "study_count/background_count" for the returned terms.

#     Notes
#     -----
#     - Tests are performed over GO terms observed in the study set. If you want a more conservative approach, iterate over all background terms instead.
#     """
#     # Study GO term counts
#     # study_terms: List[str] = [term for gene_id in study_ids
#     #                         for term in go_terms_by_id.get(gene_id, ())
#     #                         ]
#     study_term_counts  = count_go_terms(genes=study_genes, go_annotation_dict=go_terms_by_id, deduplicate_genes=True)
#     study_unique_terms = set(study_term_counts.keys())
#     if not study_unique_terms:
#         raise ValueError(F"Study IDs {study_genes} not found in background_go_terms_by_id")
#         #return {}, {}

#     #study_term_counts = Counter(study_terms)
    

#     # Background GO term counts
#     if background_term_counts == None:                              
#         background_term_counts = count_go_terms(genes = background_genes, go_annotation_dict=go_terms_by_id, deduplicate_genes=True) #allows duplicates if duplicate genes
#     background_unique_terms = list(background_term_counts.keys())
    
#     # Totals 
#     study_total = len(study_genes)
#     background_total = len(background_genes)
    
#     if background_total < study_total:
#         raise ValueError(
#             f"background_ids has fewer IDs ({background_total}) than study_ids ({study_total})."
#         )
#     background_only_total = background_total - study_total

#     # --- Fisher tests ---
#     tests: List[Tuple[str, float]] = []
#     #for term in background_unique_terms:
#     for term in study_unique_terms:
#         a = study_term_counts.get(term, 0)              # in study + has term
#         c_total = background_term_counts.get(term, 0)   # in background + has term
#         c = max(0, c_total - a)                         # in background-only + has term
        
#         # Contingency:
#         #            term     not term
#         # study        a      study_total - a
#         # bg-only      c      background_only_total - c
#         table = [
#             [a, study_total - a],
#             [c, background_only_total - c],
#         ]

#         _, p = fisher_exact(table, alternative="greater")
#         tests.append((term, p))

#     if not tests:
#         return {}, {}

#     #Multiple testing correction (BH/FDR)
#     pvals = np.array([p for _, p in tests], dtype=float)
#     adj_pvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
#     term_to_adj_p = {term: float(adj) for (term, _), adj in zip(tests, adj_pvals)}

#     # Optional filtering by adjusted p value
#     if alpha is not None:
#         term_to_adj_p = {t: p for t, p in term_to_adj_p.items() if p < alpha}

#     term_counts = {
#         term: f"{study_term_counts[term]}/{background_term_counts.get(term, 0)}"
#         for term in term_to_adj_p
#     }
    
#     if term_function:
#         terms = list(term_to_adj_p.keys())
#         term_to_function = go_ID_to_function(go_IDs=terms)
#         term_to_adj_p = {term_to_function.get(term,term):val for term, val in term_to_adj_p.items()}
#         term_counts =  {term_to_function.get(term,term):val for term, val in term_counts.items()}

#     return term_to_adj_p, term_counts

def get_gene_to_go(genes, go_annotation_dict):
    gene_to_go_terms = {gene: set(go_annotation_dict.get(gene, []))
                        for gene in genes}
    return gene_to_go_terms
    

def prepare_go_enrichment(population_genes, obodag, go_annotation_dict, alpha):
    
    gene_to_go_terms = {gene: set(go_annotation_dict.get(gene, []))
                            for gene in population_genes}
    
    return GOEnrichmentStudy(
        population_genes,
        gene_to_go_terms,
        obodag,
        propagate_counts=True,
        alpha=alpha,
        methods=["bonferroni", "fdr_bh"],
        pvalcalc="fisher_scipy_stats",
    )


def run_go_enrichment(go_model, study_genes, alpha):
    results = go_model.run_study(study_genes)

    sig = [r for r in results if r.p_fdr_bh < alpha]

    term_to_adj_p = {r.GO: r.p_fdr_bh for r in sig}
    term_counts = {r.GO: f"{r.study_count}/{r.pop_count}" for r in sig}

    return term_to_adj_p, term_counts


class DistanceMatrix:
    """
    A class for computing distance matrices, finding k-nearest neighbors (k-NN),
    and performing GO enrichment analysis for embedding-based data.
    """

    def __init__(self, embed_path, go_annotation_dict,go_obo_path ="/home/moseslab/denise/Paper/data/annotations/go_2024.obo"):  
        self.embed_df = _load_embeddings(embed_path)
        self.embed_path = embed_path      
        self.emblookup = EmbedDistanceMatrix(self.embed_df) #TODO test to see if this still works
        self.distance_m, self.distance_ids = self.emblookup.run_embedding_lookup_distance(
            queries=self.emblookup.embedding_db, metric="cosine"
        )
        
        self.id_to_idx = {id_: k for k, id_ in enumerate(self.distance_ids)}
        #self.sorted_distance_m = np.argsort(self.distance_m, axis=1)
        self.go_annotation_dict = go_annotation_dict
        
        #Background is all genes in human proteome
        #self.go_terms_background = {id.split("_")[0]: self.go_annotation_dict.get(id.split("_")[0], []) for id in self.distance_ids} #TODO change this?
        #background_genes = list(go_annotation_dict.keys())
        
        #background_genes = [_extract_gene_id(id) for id in self.distance_ids] #allows duplicates
        background_genes = list(set([_extract_gene_id(id) for id in self.distance_ids]))
        self.background_genes = background_genes
        
        #self.go_terms_background_count = get_go_term_counts(genes = background_genes, go_annotation_dict= go_annotation_dict)
        
        self.go_terms_background_count = count_go_terms(genes = background_genes, go_annotation_dict= go_annotation_dict)
        
        self.obodag = GODag(go_obo_path)
        

    def find_kNN(self, query_ids: list, k=50):
        """
        Find the k-nearest neighbors (k-NN) for a list of query IDs.

        Args:
            query_ids (list): List of query IDs to find neighbors for.
            k (int): Number of nearest neighbors to retrieve (default=50).

        Returns:
            dict: Dictionary mapping query IDs to their k-nearest neighbors.
        """
        ids = np.array(self.distance_ids)
        kNN_dict = {}
        
        #id_to_idx map
        id_to_idx = self.id_to_idx
        #df = _parse_ids(query_ids)
        
        #query ids should be gene_pos?
        present = [q for q in query_ids if q in ids]
        missing = [q for q in query_ids if not(q in ids)]
        
        if len(present) < 1:
            raise ValueError(f"None of the query IDs are present in embeddings: {self.embed_path}")
        
        elif len(missing)>1:
            print(f"Warning: {len(missing)} query IDs missing from embeddings (showing up to 10): {missing[:10]}")

        for query in present:
            idx = id_to_idx.get(query)
            if idx is None:
                continue
            
            #neighbours = self.sorted_distance_m[idx]
            row = self.distance_m[idx]
            neighbours = np.argpartition(row, k + 1)[:k + 1]
            neighbours = neighbours[neighbours != idx][:k]
            NN_genes = [_extract_gene_id(ids[idx]) for idx in neighbours] 
            kNN_dict[query] = NN_genes

        return kNN_dict

    def go_enrichment_kNN(self, query_ids:List[str], k:int = 50, alpha:float =0.05):
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
        #go_background = self.go_terms_background #TODO Change this????
        # all_ids = self.distance_ids
        # background_genes = [id.split('_')[0] for id in all_ids] #allow duplicates
        
        # background_terms= [term for id,terms in GO_ANNOTATIONS.items() for term in terms
        #     if id in background_genes]
        # background_term_counts = Counter(background_terms)
        
        kNN_dict = self.find_kNN(query_ids, k=k)

        go_enrichment_dict = {}
        go_enrichment_counts = {}
        
        go_model = prepare_go_enrichment(population_genes = self.background_genes,
                                         go_annotation_dict=self.go_annotation_dict,
                                             obodag = self.obodag,
                                             alpha=alpha)

        for query, NN_genes in kNN_dict.items():
            
            #NN_genes =  [id.split('_')[0] for id in NN_ids]
            # enriched_goterms,enriched_gocounts = go_enrichment_fisher(
            #               study_genes=NN_genes,
            #               background_genes = self.background_genes,
            #               background_term_counts = self.go_terms_background_count ,
            #               alpha=alpha,
            #               go_terms_by_id = self.go_annotation_dict)
            
            query_gene = _extract_gene_id(query)
            
            if query_gene in NN_genes: 
                NN_genes.remove(query_gene)
            
            enriched_goterms,enriched_gocounts = run_go_enrichment(go_model=go_model,
                                                                   study_genes=NN_genes,
                                                                   alpha=alpha)
            
            go_enrichment_dict[query]  = enriched_goterms#list(enriched_goterms.keys())
            go_enrichment_counts[query] = enriched_gocounts

        return go_enrichment_dict, go_enrichment_counts

    def go_enrichment_dataframe(self, query_ids:List[str], k:int = 50, alpha:float = 0.05,
                                return_function = False) -> pd.DataFrame: 
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
        go_enrichment_dict,go_enrichment_counts = self.go_enrichment_kNN(query_ids, k=k, alpha=alpha)
        
        terms = list(go_enrichment_counts.keys())
        
        #print(terms)
        
        if return_function: #Change GO ids to function labels
            go_function_dict = go_ID_to_function(go_IDs = terms)
            go_enrichment_dict = {go_function_dict.get(id,id):val for id, val in go_enrichment_dict.items()}
            go_enrichment_counts = {go_function_dict.get(id,id):val for id, val in go_enrichment_counts.items()}
        
        df_enrichment = pd.DataFrame({'term_adj_p': go_enrichment_dict, #dict
                                      'term_counts': go_enrichment_counts #dict
                                      #'term_list': terms #list
                                      }).rename_axis('query_id')
        return df_enrichment


# ----------------------------
# BLAST + random utilities
# ----------------------------

def go_enrichment_from_blast(
    blast_tsv_path: str,
    
    #go_by_gene: Mapping[str, Sequence[str]],
    go_annotation_dict: Mapping[str, Sequence[str]],
    go_obo_path:str,
    background_genes: List[str] = None,
    query_ids:List[str]=None,
    k: int = 50,
    alpha: float = 0.05,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    BLAST input format: two columns (query, hit). Both can be embedding-like IDs.
    For each query, enrich on top-k hit genes (excluding self-gene).
    """
    blast = pd.read_table(blast_tsv_path, header=None)
    blast = blast.iloc[:,:2]
    #blast.columns = ["query", "hit"] 
    blast.columns = ["query_id", "hit_id"] 

    blast["query_gene"] = blast["query_id"].astype(str).map(_extract_gene_id)
    blast["hit_gene"] = blast["hit_id"].astype(str).map(_extract_gene_id)
    
    if query_ids:
        #query_genes = [_extract_gene_id(id) for id in query_ids]
        blast = blast.loc[blast["query_id"].isin(query_ids)]
        
        if blast.empty:
            raise ValueError(f"Query ids:{query_ids[:10]} not found in blast output file (printed up to 10)")
        
    if background_genes is None:
        background_genes = list(set(blast["query_gene"]))
        
    rows = []
    obodag = GODag(go_obo_path)
    go_model = prepare_go_enrichment(population_genes = background_genes,
                                     go_annotation_dict = go_annotation_dict,
                                             obodag = obodag,
                                             alpha=alpha)
    
    for q_id, sub in blast.groupby("query_id"):
        q_gene = _extract_gene_id(q_id) 
        hits = [h for h in sub["hit_gene"].tolist() if h != q_gene][:k] #remove hits to self, remove duplicates??
        #hits = [h for h in sub["hit_id"].tolist() if h != q_id][:k]        
        term_to_adj_p, term_to_counts = run_go_enrichment(go_model=go_model,
                                                                   study_genes=hits,
                                                                   alpha=alpha)
        rows.append(
            {
                "query_id": q_id,
                "term_adj_p": term_to_adj_p,
                "term_counts": term_to_counts,
            }
        )
        
    df = pd.DataFrame(rows).set_index("query_id")

    if out_csv:
        df.to_csv(out_csv)

    return df

 
def go_enrichment_random(
    query_ids: List[str],
    background_genes: List[str],
    go_annotation_dict: Mapping[str, List[str]],
    go_obo_path:str,
    k: int = 50,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
    out_csv:str=None,
) -> pd.DataFrame:
    """
    For each query, sample k random genes from the background and run enrichment.
    Useful as a null baseline.
    """
    rng = np.random.default_rng(seed)
    bg = list(background_genes)
    if len(bg) < k:
        raise ValueError(f"background_genes size ({len(bg)}) < k ({k})")
    
    obodag = GODag(go_obo_path)
    go_model = prepare_go_enrichment(population_genes = bg,
                                     go_annotation_dict=go_annotation_dict,
                                             obodag = obodag,
                                             alpha=alpha)

    rows = []
    for q in query_ids:
        print(q)
        sample = rng.choice(bg, size=k, replace=False).tolist()
        
        term_to_adj_p, term_to_counts = run_go_enrichment(go_model=go_model,
                                                            study_genes=sample,
                                                            alpha=alpha)
        
        rows.append(
            {   "query_id": q, #make query_id the index??
                "term_adj_p": term_to_adj_p,
                "term_counts": term_to_counts,
            })
        
    df = pd.DataFrame(rows).set_index("query_id")
    if out_csv:
        df.to_csv(out_csv)
    return df

# ----------------------------
# Optional config-based loader
# ----------------------------

def load_go_from_config(config_path: str) -> Tuple[Dict[str, List[str]], str]:
    """
    Expected keys in config:
      - go_annotations: path to GO annotation tsv/csv
      - go_obo: path to .obo
    """
    cfg = load_config(config_path)
    go_annotation_dict = load_go_annotations(cfg["go_annotations"])
    obo_path = cfg["go_obo"]
    return go_annotation_dict, obo_path