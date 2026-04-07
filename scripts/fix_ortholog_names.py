#Fix ortholog names for embedding ids

#for each ortholog embedding file, read and parse names

import glob
import numpy as np

from idr_diverge.utils.helpers import embedding_h5_to_dataframe, write_h5_embed_from_df
from idr_diverge.distances.compute_ndist import _parse_ids

#If there are any names that are not in the format of "species_gene_segment_start_end", fix them
def fix_name(name, filename):
    #Example filename = '/home/moseslab/denise/embeddings/esm1b/idr_orthologs/A0A0A6YYL3_ENSG00000233917_ENSEMBL_ORTHOLOGUES_ALN_IDR_1_90_esm.h5'
    filename = filename.split('/')[-1]
    
    #if name.count('_') < 4:
    species = name.split('_')[0]
    #try to parse the name
    parts = filename.split('_')
    gene = parts[0]
    #species = parts[1]
    pos = '_'.join(parts[-4:-1])
    
    return f"{species}_{gene}_{pos}"


dir = '/home/moseslab/denise/embeddings/esm1b/idr_orthologs/'
new_dir = '/home/moseslab/denise/embeddings/esm1b/idr_orthologs_fixed_names/'

def main2():
    dir ='/home/moseslab/denise/embeddings/esm1b/idr_orthologs_fixed_names/'
    new_dir = '/home/moseslab/denise/embeddings/esm1b/idr_orthologs_fixed_names/'
    for path in glob.glob(dir + '*.h5'):
        
        ortho_df = embedding_h5_to_dataframe(path)
        all_ids = list(set(ortho_df[0]))
        ortho_ids_df = _parse_ids(all_ids)
        
        remove = ['HUMAN',np.nan]
        incorrect_names = ortho_ids_df[ortho_ids_df['__genes'].isin(remove) | 
                                       ortho_ids_df['__id'].apply(lambda x: x.count('_') < 4) 
                                       |ortho_ids_df['__gpos'].apply(lambda x: len(x.split('_')[-1])<2) ]['__id'].to_list()
        if len(incorrect_names) > 0:
            print("processing file:", path)
            corrected_names = [fix_name(name, path) for name in incorrect_names[:5]]
            print("incorrect names:", len(incorrect_names), incorrect_names[:5])
            print("corrected names:", len(corrected_names), corrected_names[:5])
            
            mask = ortho_df[0].isin(incorrect_names)
            ortho_df.loc[mask, 0] = ortho_df.loc[mask, 0].apply(lambda x: fix_name(x, path))
        
            file = path.split('/')[-1]
            outpath = f"{new_dir}{file}"
            
            #print(ortho_df[0].head())
            
            write_h5_embed_from_df(ortho_df, outpath)
        
        
def main():

    for path in glob.glob(dir + '*.h5'):
        print("processing file:", path)
        ortho_df = embedding_h5_to_dataframe(path)
        all_ids = list(set(ortho_df[0]))
        ortho_ids_df = _parse_ids(all_ids)
        
        remove = ['HUMAN',np.nan]
        incorrect_names = ortho_ids_df[ortho_ids_df['__genes'].isin(remove) | ortho_ids_df['__id'].apply(lambda x: x.count('_') < 4)]['__id'].to_list()
        
        print("incorrect names:", len(incorrect_names), incorrect_names[:5])
        
        corrected_names = [fix_name(name, path) for name in incorrect_names[:5]]
        
        print("corrected names:", len(corrected_names), corrected_names[:5])
        
        ortho_df[0] = ortho_df[0].apply(lambda x: fix_name(x, path))
        
        file = path.split('/')[-1]
        outpath = f"{new_dir}{file}"
        
        write_h5_embed_from_df(ortho_df, outpath)
   
   
if __name__ == "__main__":
    main2()
   
   
   
   

