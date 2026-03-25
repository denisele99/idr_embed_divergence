#Purpose: add columns to enrichment results:
#"terms" (list form)
#"jacc_sim_to_known"
#"bma_sim_to_known"
import pandas as pd

import sys
sys.path.append('/home/moseslab/denise/Paper')
from src.go_enrichment.go_enrich_eval import go_jaccard_similarity, go_bma_similarity
from src.go_enrichment.go_enrichment_ import GO_ANNOTATIONS, _extract_gene_id
from src.go_enrichment.go_enrichment_ import load_config,load_go_annotations,  go_ID_to_function
import argparse
import ast

def add_go_sim_columns(enrich_df, go_annotations_dict, out_path):
    enrich_df = enrich_df.copy()

    enrich_df["terms"] = enrich_df["term_counts"].apply(
        lambda x: list(ast.literal_eval(x).keys()) if pd.notna(x) else []
    )
    enrich_df["genes"] = enrich_df["query_id"].map(_extract_gene_id)

    enrich_df["jacc_sim_to_known"] = enrich_df.apply(
        lambda row: go_jaccard_similarity(
            go_annotations_dict.get(row["genes"], []),
            row["terms"],
        ),
        axis=1,
    )

    enrich_df["bma_sim_to_known"] = enrich_df.apply(
        lambda row: go_bma_similarity(
            go_annotations_dict.get(row["genes"], []),
            row["terms"],
        ),
        axis=1,
    )

    enrich_df.to_csv(out_path, index=False)
    return enrich_df


#CONFIG_PATH = "/home/moseslab/denise/Paper/configs/go_config.txt"
#config = load_config(CONFIG_PATH)

#GO_ANNOTATIONS = load_go_annotations(config["go_annotations"])

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Run similarity metrics for GO enrichment results.")

    parser.add_argument("--enrich_path",type=str, required=True, help="Path to go enrichment results table (csv).")

    parser.add_argument("--out_path", type=str, required=True, default='./out.csv',
        help="Output path (csv)")
    
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    enrich_path = args.enrich_path
    #out_path = args.out_path
    
    enrich_df = pd.read_csv(enrich_path)
    
    add_go_sim_columns(
        enrich_df=enrich_df,
        go_annotations_dict=GO_ANNOTATIONS,
        out_path=args.out_path,
    )

    

if __name__ == "__main__":
    main()

