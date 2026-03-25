
- embeddings are read in hd5 format OR csv
- for all embedding files IDs must always be in the first column (labelled 0)
- must be format SPECIES/ENSEMBLID_GENE_SEGMENT_start_end (e.g. ____)
- for human ids, the beginning of the id is labelled HUMAN_GENE_SEGMENT_start_end (e.g __)
-must be all caps 
-dependencies

Ortholog file names:
- format should be ENSEMBLID_GENE_etc.
- can put orthologs in one directory and use as input, or combine all ortholog embeddings into one file