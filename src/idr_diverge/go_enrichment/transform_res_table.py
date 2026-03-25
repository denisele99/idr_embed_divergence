
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

CONFIG_PATH = "/home/moseslab/denise/Paper/configs/go_config.txt"
config = load_config(CONFIG_PATH)

GO_ANNOTATIONS = load_go_annotations(config["go_annotations"])


enrich_01 = pd.read_csv('/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_idr_bg_study_unique.csv')
enrich_005 = pd.read_csv('/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_idr_bg_study_unique_005.csv')

enrich_005.rename(columns={'Unnamed: 0': 'query_id', 'enriched_terms':"term_adj_p"}, inplace=True)
enrich_01.rename(columns={'Unnamed: 0': 'query_id', 'enriched_terms':"term_adj_p"}, inplace=True)


genes = list(set(enrich_01['query_id'].apply(lambda x: x.split('_')[0])))
#gene_to_terms = {g:GO_ANNOTATIONS.get(g,0) for g in genes if not(GO_ANNOTATIONS.get(g,0) == 0)}

enrich_01['terms'] = enrich_01['term_counts'].apply(lambda x: list(eval(x).keys()))
enrich_01['genes'] = enrich_01['query_id'].map(_extract_gene_id)

enrich_01["jacc_sim_to_known"] = enrich_01.apply(lambda row: go_jaccard_similarity(
                                                    GO_ANNOTATIONS.get(row['genes'],[]),
                                                    row['terms']),axis=1)

enrich_01["bma_sim_to_known"] = enrich_01.apply(lambda row: go_bma_similarity(
                                                    GO_ANNOTATIONS.get(row['genes'],[]),
                                                    row['terms']),axis=1)


enrich_01.to_csv('/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_idr_bg_study_unique_001_eval.csv', index=False)


enrich_005['terms'] = enrich_005['term_counts'].apply(lambda x: list(eval(x).keys()))
enrich_005['genes'] = enrich_005['query_id'].map(_extract_gene_id)

enrich_005["jacc_sim_to_known"] = enrich_005.apply(lambda row: go_jaccard_similarity(
                                                        GO_ANNOTATIONS.get(row['genes'],[]),
                                                        row['terms']),axis=1)

enrich_005["bma_sim_to_known"] = enrich_005.apply(lambda row: go_bma_similarity(
                                                        GO_ANNOTATIONS.get(row['genes'],[]),
                                                        row['terms']),axis=1)

enrich_005.to_csv('/home/moseslab/denise/Paper/res/go_enrich/knn_enrichment_idr_bg_study_unique_005_eval.csv', index=False)