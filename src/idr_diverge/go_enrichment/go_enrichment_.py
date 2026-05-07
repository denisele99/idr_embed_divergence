from collections import Counter
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Mapping, Sequence, Optional
from collections import defaultdict, Counter
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
#from pygosemsim import graph,similarity
from pygosemsim import graph as go_graph, similarity
from functools import lru_cache
from pathlib import Path


from idr_diverge.distances.embed_distance import EmbedDistanceMatrix
from idr_diverge.utils.helpers import load_config, resolve_config_paths
from idr_diverge.distances.compute_ndist import  _load_embeddings


#This module was written by the Denise Le, with some functions edited and refactored with assistance from ChatGPT (OpenAI) for clarity and maintainability. 
#All changes were reviewed and validated by the author.

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "go_config.yaml"
)

print(DEFAULT_CONFIG_PATH)


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
    go_annot = read_go_annotations(annot_df.dropna(subset=['Gene Ontology IDs']))
    return go_annot


config_data = load_config(Path(DEFAULT_CONFIG_PATH))

config_data = resolve_config_paths(config=config_data, config_path=str(DEFAULT_CONFIG_PATH),
                                  path_keys={'go_annotations', 'go_obo'})


GO_ANNOTATIONS = load_go_annotations(config_data.get("go_annotations"))
OBO_PATH = str(config_data["go_obo"])


@lru_cache(maxsize=1)
def load_go_sim_graph(obo_path:str = OBO_PATH):
    print(obo_path.split('.obo')[0])
    G = go_graph.from_resource(obo_path.split('.obo')[0])
    similarity.precalc_lower_bounds(G)
    return G

GO_GRAPH = load_go_sim_graph()

def go_ID_to_function(go_IDs:list,graph=GO_GRAPH) -> Dict[str, str]:
    '''
    Converts list of GO IDs to associated function
        input: list of GO IDs
        returns: dict -> {GO ID: function;...} 
    '''
    functions = {ID_: graph.nodes[ID_]['name'] for ID_ in go_IDs if ID_ in graph.nodes}
    
    return functions


def _extract_gene_id(_id: str, *, delimiter: str = "_") -> str:
    """
    Extract the UniProt/gene identifier from an embedding ID.

    Expected format:
        SPECIES_GENE_SEGTYPE_START_END or GENE_SEGTYPE_START_END
    Example:
        HUMAN_Q14004_IDR_1_694 -> Q14004
    """
    parts = str(_id).split(delimiter)
    if len(parts) < 5:
        return parts[0]
    return parts[1]

def count_go_terms(
    genes: List[str],
    go_annotation_dict: Dict[str, List[str]],
    deduplicate_genes: bool = False,
) -> Counter:
    """
    Count GO terms across a gene set.

    If deduplicate_genes=True, repeated genes are counted once.
    """
    gene_list = list(genes)
    if deduplicate_genes:
        gene_list = list(set(gene_list))  # stable unique

    terms: List[str] = []
    for g in gene_list:
        terms.extend(go_annotation_dict.get(g, ()))
    return Counter(terms)

    

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
        
        self.go_terms_background_count = count_go_terms(genes = background_genes, go_annotation_dict= go_annotation_dict)
        
        self.go_obo_path = go_obo_path
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
        
        elif len(missing)>0:
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
        
        kNN_dict = self.find_kNN(query_ids, k=k)

        go_enrichment_dict = {}
        go_enrichment_counts = {}
        
        go_model = prepare_go_enrichment(population_genes = self.background_genes,
                                         go_annotation_dict=self.go_annotation_dict,
                                             obodag = self.obodag,
                                             alpha=alpha)

        for query, NN_genes in kNN_dict.items():
            
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
        
        if return_function: #Change GO ids to function labels
            GO_GRAPH = load_go_sim_graph(str(self.go_obo_path))
            all_go_ids = set()
            for term_dict in go_enrichment_dict.values():
                all_go_ids.update(term_dict.keys())
            for term_dict in go_enrichment_counts.values():
                all_go_ids.update(term_dict.keys())

            go_function_dict = go_ID_to_function(list(all_go_ids))

            go_enrichment_dict = {
                query_id: {go_function_dict.get(go_id, go_id): val for go_id, val in term_dict.items()}
                for query_id, term_dict in go_enrichment_dict.items()
            }
            go_enrichment_counts = {
                query_id: {go_function_dict.get(go_id, go_id): val for go_id, val in term_dict.items()}
                for query_id, term_dict in go_enrichment_counts.items()
            }

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
    null baseline.
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

