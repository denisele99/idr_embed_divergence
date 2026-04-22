import sys
sys.path.append('/home/moseslab/denise/Paper/src')
from distances.neighbour_divergence import run_fam_divergence_pipeline
from utils.helpers import load_config


#CONFIG_PATH = '/home/moseslab/denise/Paper/configs/go_config.txt'
CONFIG_PATH = '/home/moseslab/denise/Paper/src/distances/config_FND.yaml'
CONFIG_DATA = load_config(CONFIG_PATH)


def main():
    
    input_fam_ids = ['1158']#'638,1849,627,368,756,1834,685,1136,2303,20'.split(',')
    #run_fam_divergence_pipeline()
    run_fam_divergence_pipeline(
    fam_ids=input_fam_ids,
    out_dir='/home/moseslab/denise/Paper/data/neighbour_distance_matrix',
    dist_type="between",
    segment="idr",
    human_embed_path=CONFIG_DATA['human_idr_embed_path'],
    ortho_dir=CONFIG_DATA['idr_ortho_embed_dir'],
    ortholog_filter=False,
    precompute_orthologs=True
    )
    

if __name__  == "__main__":
    main()