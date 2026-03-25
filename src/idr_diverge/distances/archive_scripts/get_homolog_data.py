import sys
import numpy as np
import glob
import pandas as pd
sys.path.append('/home/moseslab/denise/Thesis/src')
from utils.helper_functions import read_pickle, save_pickle


import requests
from typing import Optional, Dict, Any, List, Tuple, Iterable
from pathlib import Path

#Edited 10-10-25

#Get homology information (one2one orthologs) from ensembl based on human uniprot ids

def get_ensembl_homology_json(
    gene_id: str,
    homology_type: str = "orthologues",
    source_species:str = 'human',
    target_species: Optional[str] = None,
    sequence: str = "none",
    aligned: bool = False,
    server: str = "https://rest.ensembl.org"
) -> Optional[Dict[str, Any]]:
    """
    Retrieve homology data for a given Ensembl gene ID from the Ensembl REST API.

    Parameters
    ----------
    gene_id : str
        Ensembl gene ID (e.g., 'ENSG00000174939').
    type_ : str, optional
        Type of homology to retrieve: 'orthologues', 'paralogues', or 'all'. Default is 'orthologues'.
    target_species : str, optional
        Comma-separated list of target species (e.g., 'mus_musculus,cavia_porcellus').
        If None, retrieves all available species.
    sequence : str, optional
        Type of sequence data to include: 'none', 'cdna', 'protein', or 'genomic'. Default is 'none'.
    aligned : bool, optional
        Whether to include sequence alignments. Default is False.
    server : str, optional
        Base URL of the Ensembl REST API. Default is 'https://rest.ensembl.org'.

    Returns
    -------
    dict or None
        Parsed JSON response containing homology data, or None if the request failed.
    """
    endpoint = f"/homology/id/{source_species}/{gene_id}"
    url = f"{server}{endpoint}"

    params = {
        "type": homology_type,
        "sequence": sequence,
        "aligned": int(aligned)
    }
    if target_species:
        params["target_species"] = target_species

    try:
        r = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {gene_id}: {e}")
        return None


def _get(d: Dict[str, Any], *keys, default=None):
    """Return the first present key in d among *keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _normalize_homology(h: Dict[str, Any]) -> Dict[str, Any]:
    # Some responses use h['species'], h['id'], h['percent_id'], others use h['target'][...]
    t = h.get("target", {})
    return {
        "rel_type": _get(h, "type"),
        "taxon_level": _get(h, "taxonomy_level"),
        "target_species": _get(t, "species", "target_species", default=_get(h, "species")),
        "target_gene_id": _get(t, "id", "target_id", default=_get(h, "id")),
        #"target_gene_id2": _get(t, "target","id",default=_get(h, "id")),
        "target_protein_id": _get(t,"target","protein_id", default=_get(h, "protein_id")),
        # identity fields show up with slightly different names across releases
        "percent_id": _get(h, "percent_id", "perc_id", default=_get(t, "percent_id", "perc_id")),
       # "goc_score": _get(h, "goc_score", default=_get(t, "goc_score")),
       # "wga_coverage": _get(h, "wga_coverage", default=_get(t, "wga_coverage")),
       # "dn": _get(h, "dn", default=_get(t, "dn")),
       # "ds": _get(h, "ds", default=_get(t, "ds")),
       # "is_tree_compliant": _get(h, "is_tree_compliant", default=_get(t, "is_tree_compliant")),
    }

def parse_homologies(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # payload['data'] is a list; each entry corresponds to a source gene
    result = []
    for entry in payload.get("data", []):
        src_gene = entry.get("id")
        for h in entry.get("homologies", []):
            row = _normalize_homology(h)
            row["source_gene_id"] = src_gene
            result.append(row)
    return result



def load_uniprot_ensembl_maps(
    mapping_tsv: str | Path = '/home/moseslab/denise/IDR_LM/data/annotations/uniprot_ensembl_idmapping_2025_10_10.tsv',
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load UniProt ↔ Ensembl gene ID mappings.

    Returns
    -------
    uniprot_to_ensembl : dict
        UniProt ID → Ensembl gene ID (version stripped)
    ensembl_to_uniprot : dict
        Ensembl gene ID → UniProt ID
    """
    df = pd.read_table(mapping_tsv, usecols=["From", "To"])

    # UniProt → list of Ensembl gene IDs
    grouped = df.groupby("From")["To"].apply(sorted)

    # choose first (latest) and strip version suffix
    uniprot_to_ensembl = {
        uniprot: genes[0].split(".", 1)[0]
        for uniprot, genes in grouped.items()
    }
    
    # uniprot_to_ensembl_gene = {uniprot:gene_ids[0].split('.')[0] for uniprot,gene_ids in uniprot_to_ensembl_gene.items()}

    ensembl_to_uniprot = {ens: uni for uni, ens in uniprot_to_ensembl.items()}

    return uniprot_to_ensembl, ensembl_to_uniprot

def uniprot_to_ensembl(uniprot_ids:list,
                       uniprot_ensembl_dict = load_uniprot_ensembl_maps()[0]):
    '''Uniprot to ensembl GENE id'''
    ensembl_ids = [uniprot_ensembl_dict.get(id) for id in uniprot_ids if id in uniprot_ensembl_dict.keys()]
    return ensembl_ids

def ensembl_to_uniprot(ensembl_ids:list,
                       ensembl_uniprot_dict = load_uniprot_ensembl_maps()[1]):
    #Ensembl GENE id to uniprot
    uniprot_ids = [ensembl_uniprot_dict .get(id) for id in ensembl_ids if id in ensembl_uniprot_dict.keys()]
    return uniprot_ids


#Download and parse homology data for a list of uniprot ids

def main():
    from pathlib import Path
    import argparse
    
    p = argparse.ArgumentParser(description="Get homology information from ensembl, by human uniprot id search.")
    
    p.add_argument("--ids", nargs="+", help='Uniprot ids: --ids a b c')                        # 1+ values: --ids a b c
    p.add_argument("--path", type=Path, help='outpath')   
    p.add_argument("--homology-type", type=str, choices=['orthologues', 'paralogues', 'all'], 
                   default = 'orthologues',
                   help='Type of homology to filter by: orthologues, paralogues, or all. Default is orthologues')   
    p.add_argument("--source-spp", type=str, default='human', help='Species of the input ids')  
    
    # p.add_argument("--lr", type=float, required=True)         # required option
    # p.add_argument("--species", choices=["human","mouse"])    # enum-like
    
    # p.add_argument("--pairs", nargs=2, metavar=("A","B"))     # exactly 2 values
    # p.add_argument("--flag", action="store_true")             # boolean flag
    # p.add_argument("--kv", action="append")                   # repeatable: --kv a=1 --kv b=2
    #                     # auto-validates filesystem paths
    
    args = p.parse_args()
    
    ensembl_gene_ids = uniprot_to_ensembl(args.ids)
    out_path = args.path
    homology=args.homology_type
    source_spp =args.source_spp
    
    out_data =[]
    
    #Convert gene id to ensembl ids
    
    if ensembl_gene_ids:
        for gene_id in ensembl_gene_ids:
            print(gene_id,'...')
            data = get_ensembl_homology_json(gene_id=gene_id,
                                            homology_type=homology,
                                            source_species=source_spp)
            rows = parse_homologies(data)
            
            out_data.append(rows)
        #write to pickle
        print('Writing to ', out_path)
        save_pickle(data=out_data,outpath=out_path)
    else:
        print(f"No ensembl gene ids found for ids: {args.ids}")

if __name__ =="__main__":
    main()

#Get homology data summary for ortholog filtering

#From filter_orthologs file

def get_homology_data_summmary():
    paths = glob.glob('/home/moseslab/denise/Thesis/results/annotations/homology_labels_2/*')
    genelist = [path.split('/')[-1].split('_')[0] for path in paths]
    
    df_ls= []
    for gene in genelist:
        print(gene)
        path = f'/home/moseslab/denise/Thesis/results/annotations/homology_labels_2/{gene}_homol.pkl'
        try:
            homol_file = read_pickle(path)
            homol_df = pd.DataFrame([d for sublist in homol_file for d in sublist])
            #df_ls.append(homol_df[homol_df['rel_type'] == 'ortholog_one2one'])
            df_ls.append(homol_df[(homol_df['rel_type'] == 'ortholog_one2one') | (homol_df['rel_type'] == 'ortholog_one2many')])
        except Exception as e:
            print(f'No homology data for {gene}: {e}')
            continue

    homol_df_combined = pd.concat(df_ls, ignore_index=True)
    homol_df_combined['spp_prefix'] = homol_df_combined['target_protein_id'].apply(lambda x: ''.join(re.findall(r'[A-Za-z_]', x)))
    
    s = homol_df_combined.groupby('rel_type')['spp_prefix'].value_counts()
    spp_counts_nested = {homol: prefix.droplevel('rel_type').astype(int).to_dict()
            for homol, prefix in s.groupby(level='rel_type')}

    ensembl_protein_dict= homol_df_combined.groupby('rel_type')['target_protein_id'].apply(list).to_dict()
    percent_id_dict = homol_df_combined.set_index('target_protein_id')['percent_id'].to_dict()

    ensembl_homol_reference = {'ensembl_protein_orthologs':ensembl_protein_dict,
                            'spp_prefix_counts':spp_counts_nested,
                            'percent_ids':percent_id_dict}

    #Save as pickle
    save_pickle(data=ensembl_homol_reference, 
                outpath='/home/moseslab/denise/Thesis/results/annotations/ensembl_homol_reference_10-10')
    
#ensembl_homol_reference = read_pickle('/home/moseslab/denise/Thesis/results/annotations/ensembl_homol_reference_10-10.pkl')
#percent_id_dict = ensembl_homol_reference['percent_ids']