print('Getting GO annotations...')

go_obo = GeneOntology(go_obo_path)
yeast_annot = pd.read_csv('/home/moseslab/denise/IDR_LM/data/annotations/yeast_goids_2023_filtered.csv')
go_annotations = read_go_annotations(yeast_annot)

print('Getting go obo...')
go_obo_path = "/home/moseslab/denise/IDR_LM/data/annotations/go_2024.obo"

#From /home/moseslab/denise/Paper/src/go_enrichment/go_enrichment_analysis.py

if os.path.exists(go_obo_path):
    graph = obonet.read_obo(go_obo_path)
else:
    url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    graph = obonet.read_obo(url)


#From /home/moseslab/denise/Paper/src/go_enrichment/go_enrichment.py

#SET GLOBAL VARIABLES HERE
go_background =''

#TODO set go_annotation dict as a global variable?
annot_df = pd.read_table('/home/moseslab/denise/IDR_LM/data/annotations/idr_annotations/uniprotkb_Human_AND_model_organism_9606_2024_08_20.tsv')
go_annotation_dict = read_go_annotations(annot_df.dropna(subset='Gene Ontology IDs'))

G = graph.from_resource("/home/moseslab/denise/IDR_LM/data/annotations/go_2024")
similarity.precalc_lower_bounds(G)