


import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Any

from idr_diverge.go_enrichment.go_enrichment_ import DistanceMatrix,go_enrichment_from_blast,go_enrichment_random, load_go_annotations, _extract_gene_id
from idr_diverge.distances.compute_ndist import _load_embeddings
from idr_diverge.utils.helpers import load_config, resolve_config_paths

"""
Portions of this script were generated or refined with assistance from ChatGPT (OpenAI) and have been reviewed and modified for this project by the author.
"""

def parse_query_ids(value: List[str]) -> List[str]:
    """Parse comma-separated IDs from CLI."""
    ids = [x.strip() for x in value.split(",") if x.strip()]
    return ids

def get_param(args, config: Dict[str, Any], key: str, default=None):
    """Use CLI value if provided, otherwise fall back to config, then default."""
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GO enrichment for knn, BLAST, or RANDOM search_methods.")

    parser.add_argument("--config",type=str, required=True, help="Path to YAML/TXT config file.")

    parser.add_argument("--dataset", type=str, default=None,
        help="Input dataset path. For knn/RANDOM: embedding file. For BLAST: BLAST output file."
    )
    parser.add_argument("--query_ids", type=str, default=None,
        help="Comma-separated query IDs. If omitted, uses config query_ids or all targets."
    )
    parser.add_argument("--k", type=int,default=None,
        help="Number of nearest neighbors / hits to consider."
    )
    parser.add_argument("--output_path", type=str,default=None,
        help="Path to save output CSV."
    )
    parser.add_argument("--search_method", type=str, choices=["knn", "BLAST", "RANDOM"], default=None,
        help="Which enrichment mode to run."
    )
    parser.add_argument("--alpha", type=float, default=None,
        help="Adjusted p-value cutoff."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    config = load_config(args.config)
    config = resolve_config_paths(config=config, config_path=args.config,
                                  path_keys={'go_annotations', 'go_obo', 'target_embeddings', 'target_blast_output', 'output'})
    print(config)

    search_method = get_param(args, config, "search_method", "knn")
    k = get_param(args, config, "k", 50)
    alpha = get_param(args, config, "alpha", 0.01)
    
    output_path = get_param(args, config, "output", "go_enrichment_result.csv")
    
    go_annotation_path = config.get("go_annotations")
    go_obo_path = config.get("go_obo")
    target_embeddings = config.get("target_embeddings")
    target_blast_output = config.get("target_blast_output")
    
    query_ids = get_param(args, config, "query_ids", None)
    query_all = config.get("query_all", True)
    
    
    if query_ids not in (None, "null"):
        query_ids = parse_query_ids(query_ids)
        print("query ids:", query_ids)
    elif not query_all and query_ids is None:
        raise ValueError("No query IDs provided and query_all is False.")
    #else:
    #    query_all = True
    
    
    if go_obo_path is None:
        raise ValueError("Config must contain 'go_obo' path.")

    go_annotation_dict = load_go_annotations(go_annotation_path)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if search_method == "knn":
        dataset_path = args.dataset or target_embeddings
        if dataset_path is None:
            raise ValueError("knn mode requires 'target_embeddings' in config or --dataset.")

        enrich = DistanceMatrix(dataset_path, go_annotation_dict,go_obo_path)
        if query_all or query_ids is None:
            query_ids = list(enrich.distance_ids)
  
        enrich_df = enrich.go_enrichment_dataframe(
            query_ids=query_ids,
            k=k,
            alpha=alpha,
        )
        enrich_df.to_csv(output_path, index=True)

    elif search_method == "BLAST":
        blast_path = args.dataset or target_blast_output
        if blast_path is None:
            raise ValueError("BLAST mode requires 'target_blast_output' in config or --dataset.")

        if query_ids is None or query_all:
            blast = pd.read_table(blast_path, header=None)
            query_ids = list(blast[0].unique())
        
        #Background should be same as genes/ids found in embedding background
        dataset_path = target_embeddings
        distance_ids = list(_load_embeddings(target_embeddings).iloc[:,0]) #TODO is this correct?
        background_genes =list(set([_extract_gene_id(id) for id in distance_ids])) #no duplicates
        
        go_enrichment_from_blast(blast_tsv_path=blast_path,
            query_ids = query_ids,
            background_genes = background_genes,
            k=k,
            alpha=alpha,
            out_csv=str(output_path),
            go_annotation_dict=go_annotation_dict,
            go_obo_path = go_obo_path
        )

    elif search_method == "RANDOM":
        dataset_path = args.dataset or target_embeddings
        if dataset_path is None:
            raise ValueError("RANDOM mode requires 'target_embeddings' in config or --dataset.")

        all_ids = list(_load_embeddings(dataset_path).iloc[:,0])
        background_genes = list(set([_extract_gene_id(id) for id in all_ids]))
        #background_genes = list(go_annotation_dict.keys())

        if query_all or query_ids is None:
            query_ids = all_ids
        
        enrich_df = go_enrichment_random(
            query_ids=query_ids,
            background_genes=background_genes,
            k=k,
            alpha=alpha,
            go_annotation_dict=go_annotation_dict,
            go_obo_path=go_obo_path
        )
        enrich_df.to_csv(output_path, index=True)

    else:
        raise ValueError(f"Unrecognized search_method: {search_method}")

    print(f"Saved GO enrichment results to: {output_path}")

if __name__ == "__main__":
    main()