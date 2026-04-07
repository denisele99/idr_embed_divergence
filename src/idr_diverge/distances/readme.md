
- embeddings are read in hd5 format OR csv
- for all embedding files IDs must always be in the first column (labelled 0)
- must be format SPECIES/ENSEMBLID_GENE_SEGMENT_start_end (e.g. ____)
- for human ids, the beginning of the id is labelled HUMAN_GENE_SEGMENT_start_end (e.g __)
-must be all caps 
-dependencies

Ortholog file names:
- format should be ENSEMBLID_GENE_etc.
- can put orthologs in one directory and use as input, or combine all ortholog embeddings into one file

Requirements for ortholog homolog_annotation file
- Downloaded from (?)
- saved as pickle dict with the following format:
'ensembl_protein_orthologs', 'spp_prefix_counts', 'percent_ids'

{'ensembl_protein_orthologs': {'ortholog_one2many': ['ENSECRP00000021239','ENSECRP00000020241',],
                                'ortholog_one2one': ['ENSNLEP00000045841','ENSPTRP00000029753',]},
'spp_prefix_counts': {'ortholog_one2many': {'ENSCARP': 21335, 'ENSHHUP': 18466},
                        ortholog_one2one': {'ENSCARP': 21335, 'ENSHHUP': 18466} },
'percent_ids': {'ENSNLEP00000045841': 91.1677, 'ENSPTRP00000029753': 91.5916 ...}
}