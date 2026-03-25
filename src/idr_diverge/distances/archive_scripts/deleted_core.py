
def filter_orthologs(
    ortholog_ids: Iterable[str],
    ensembl_homol_reference: Dict = read_pickle(CONFIG_DATA['ensembl_homol_ref']),
    ortholog_type: Literal["ortholog_one2one", "ortholog_one2many", "ortholog_one2one_one2many",] = "ortholog_one2one",
    species_threshold: float = 0.5,
    drop_duplicates: Literal["random", "percent_id"] = "percent_id",
    #spp_count_proteome_threshold: int = 1000,
    n_orthologs:int=10,
) -> List[str]:
    """
    Filter a batch of ortholog region IDs given homology references and a per-protein %identity map.
    
    Args:
        ortholog_ids: Iterable of ortholog region ID strings.
        ensembl_homol_reference: Dictionary containing homology reference data.
        ortholog_type: Type of ortholog relationship to filter by.
        species_threshold: Minimum fraction of orthologous segments in a family a species must cover.
        drop_duplicates: Method to use for deduplication ('random' or 'percent_id').
        ##spp_count_proteome_threshold: Minimum proteome count threshold for species inclusion.
        n_orthologs: Minimum number of unique ortholog species required after filtering.

    Steps
    -----
    1) Keep only rows whose Ensembl protein is in allowed orthologs of `ortholog_type`.
    2) Keep only species whose total proteome count for this `ortholog_type` exceeds `spp_count_proteome_threshold`,
       plus always keep HUMAN.
    3) Deduplicate by (__species, __gpos):
         - If drop_duplicates == 'percent_id': keep the row with the highest percent_id
           (from percent_id_dict, missing treated as -inf).
         - If drop_duplicates == 'random': pick one row at random per group (stable given random_state).
    4) Enforce per-species coverage: drop species that appear in fewer than
       ceil(n_unique_gene_pos * species_threshold) unique gene positions.
    5) Return the original IDs (column '__ids') as a list.
    """

    ensembl_homol_reference = read_pickle(CONFIG_DATA['ensembl_homol_ref'])
       
    # ---- Parse ortholog IDs ----
    parsed = _parse_ids(ortholog_ids)  # expects columns: __ensembl_id, __species, __gpos, __ids
    
    if parsed.empty:
        return []

    df = parsed.copy()
    n_members = df["__gpos"].nunique()

    # ---- (1) Allowed ortholog proteins for this ortholog type ----
    prot_map = ensembl_homol_reference.get("ensembl_protein_orthologs", {})
    if ortholog_type == "ortholog_one2one_one2many":
        allowed_protein_ids = set(prot_map.get("ortholog_one2one", []) + prot_map.get("ortholog_one2many", []))
    else:
        allowed_protein_ids = set(prot_map.get(ortholog_type, []))
    if not allowed_protein_ids: # Nothing allowed → early exit
        return []

    # ---- (2) Allowed species by proteome counts + always HUMAN ----
    spp_counts_by_type = ensembl_homol_reference.get("spp_prefix_counts", {})
    if ortholog_type == "ortholog_one2one_one2many":
        spp_counts_all = spp_counts_by_type.get("ortholog_one2one", {}) | spp_counts_by_type.get("ortholog_one2many", {})
    else:
        spp_counts_all = spp_counts_by_type.get(ortholog_type, {}) or {}
    spp_counts_filtered = {
        spp: int(cnt) for spp, cnt in spp_counts_all.items()
        if pd.notna(cnt) #and int(cnt) > spp_count_proteome_threshold
    }
    allowed_spp_prefixes = set(spp_counts_filtered) | {"HUMAN"}

    # Apply filters 1 & 2
    df = df[(df["__ensembl_id"].isin(allowed_protein_ids) & df["__species"].isin(allowed_spp_prefixes)) | (df["__species"]=='HUMAN')].copy() 
    if df.empty:
        return []    
    
    # (3) Exclude duplicates within (__species, __gpos)
    duplicates = df[df.duplicated(subset=['__species','__gpos'],keep=False)]
    
    if duplicates.any().any():
        if drop_duplicates=='percent_id':
            percent_id_dict = ensembl_homol_reference['percent_ids']
            
            duplicates['percent_id'] = duplicates['__ensembl_id'].apply(lambda x: percent_id_dict.get(x, None))
            #add to filtered ids
            df = pd.concat([duplicates,df])
            #drop duplicates with lower percent id
            df = df.sort_values(by=['__species','__gpos','percent_id'], ascending=[True,True,False]).drop_duplicates(subset=['__species','__gpos'], keep="first")
        elif drop_duplicates=='random':
            df = df.drop_duplicates(subset=['__species','__gpos'], keep="first")
        else:
            raise ValueError("drop_duplicates must be 'percent_id' or 'random'")

    if df.empty:
        return []

    if species_threshold:
        # ---- (4) Enforce species coverage threshold ----
        # Exclude species prefixes with counts < coverage threshold, relative to total n members (idrs in family)
        # require each species to cover at least ceil(n_members * species_threshold) distinct gene positions
        min_required = math.ceil(n_members * float(species_threshold))

        # compute per-species unique __gpos counts, align back
        spp_unique_counts = df.groupby("__species")["__gpos"].nunique()       
        mask = spp_unique_counts >=min_required
        df = df[df['__species'].isin(set(spp_unique_counts[mask].index))]
        if df.empty:
            return []
        
    
    #TODO ADD FILTER FOR N ORTHOLOGS per fam (at least 50?)
    if len(set(df['__species'])) < n_orthologs:
        return []

    #print(len(df['__species'].unique()))
    # ---- (5) Return the filtere


def filter_orthologs_2(
    ortholog_ids: Iterable[str],
    ensembl_homol_reference: Dict = read_pickle(CONFIG_DATA['ensembl_homol_ref']),
    ortholog_type: Literal["ortholog_one2one", "ortholog_one2many", "ortholog_one2one_one2many",] = "ortholog_one2one",
    species_threshold: float = 0.5,
    n_orthologs:int=10,
) -> List[str]:
    """
    Filter a batch of ortholog region IDs given homology references and a per-protein %identity map.
    
    Args:
        ortholog_ids: Iterable of ortholog region ID strings. Expects format: EnsemblID_gene_segment_position
        ensembl_homol_reference: Dictionary containing homology reference data.
        ortholog_type: Type of ortholog relationship to filter by.
        species_threshold: Minimum fraction of orthologous segments in a family a species must cover.
        ##drop_duplicates: Method to use for deduplication ('random' or 'percent_id').
        ##spp_count_proteome_threshold: Minimum proteome count threshold for species inclusion.
        n_orthologs: Minimum number of unique ortholog species required after filtering.

    Steps
    -----
    1) Keep only rows whose Ensembl protein is in allowed orthologs of `ortholog_type`.
    2) Keep only species whose total proteome count for this `ortholog_type` exceeds `spp_count_proteome_threshold`,
       plus always keep HUMAN.
    3) Deduplicate by (__species, __gpos):
         - If drop_duplicates == 'percent_id': keep the row with the highest percent_id
           (from percent_id_dict, missing treated as -inf).
         - If drop_duplicates == 'random': pick one row at random per group (stable given random_state).
    4) Enforce per-species coverage: drop species that appear in fewer than
       ceil(n_unique_gene_pos * species_threshold) unique gene positions.
    5) Return the original IDs (column '__ids') as a list.
    """
       
    # ---- Parse ortholog IDs ----
    parsed = _parse_ids(ortholog_ids)  # expects columns: __ensembl_id, __species, __gpos, __ids
    
    if parsed.empty:
        return []

    df = parsed.copy()
    n_members = df["__gpos"].nunique()

    # ---- (1) Allowed ortholog proteins for this ortholog type ----
    prot_map = ensembl_homol_reference.get("ensembl_protein_orthologs", {})
    if ortholog_type == "ortholog_one2one_one2many":
        allowed_protein_ids = set(prot_map.get("ortholog_one2one", []) + prot_map.get("ortholog_one2many", []))
    else:
        allowed_protein_ids = set(prot_map.get(ortholog_type, []))
    if not allowed_protein_ids: # Nothing allowed → early exit
        return []
    
    # Apply filters 1 & 2
    df = df[df["__ensembl_id"].isin(allowed_protein_ids) | (df["__species"]=='HUMAN')].copy() 
    if df.empty:
        return []    
    
    # (3) Exclude duplicates within (__species, __gpos)
    duplicates = df[df.duplicated(subset=['__species','__gpos'],keep=False)]
    
    if duplicates.any().any():
        percent_id_dict = ensembl_homol_reference['percent_ids']
        duplicates['percent_id'] = duplicates['__ensembl_id'].apply(lambda x: percent_id_dict.get(x, None))
        #add to filtered ids
        df = pd.concat([duplicates,df])
         # Randomize order to break ties randomly
        df = df.sample(frac=1).reset_index(drop=True)
        
         # Prioritize highest percent identity. If percent identity not available, drop random duplicate
        df = (
            df.sort_values(
                ['__species','__gpos','percent_id'],
                ascending=[True,True,False],
                na_position='last'
            )
            .drop_duplicates(['__species','__gpos'], keep='first')
        )
    
    if species_threshold:
        # ---- (4) Enforce species coverage threshold ----
        # Exclude species prefixes with counts < coverage threshold, relative to total n members (idrs in family)
        # require each species to cover at least ceil(n_members * species_threshold) distinct gene positions
        min_required = math.ceil(n_members * float(species_threshold))

        # compute per-species unique __gpos counts, align back
        spp_unique_counts = df.groupby("__species")["__gpos"].nunique()       
        mask = spp_unique_counts >=min_required
        df = df[df['__species'].isin(set(spp_unique_counts[mask].index))]
        if df.empty:
            return []
        
    
    #TODO ADD FILTER FOR N ORTHOLOGS per fam (at least 50?)
    if len(set(df['__species'])) < n_orthologs:
        return []

    #print(len(df['__species'].unique()))
    # ---- (5) Return the filtere


#Filter orthologs after calculating distance matrix

def filter_dist_matrix():
    pass

from typing import Dict, Iterable, List, Literal, Optional
def filter_dist_by_orthologs(ortho_ids, dist_res:Dict,
                            ortholog_type = "ortholog_one2one", 
                            species_threshold = 0.5,
                            drop_duplicates = "percent_id",
                            n_orthologs=10):
    '''
    filter dist_matrix (dist_res by its orthologs)
    
    :param ortho_ids: Description
    :param dist_res: Description
    :type dist_res: Dict
    :param ortholog_type: Description
    :param species_threshold: Description
    :param drop_duplicates: Description
    :param n_orthologs: Description
    '''
    #dist_res = Dict of distances for a family
    #ortho_ids = list of full ortholog ids for that family (spp_gene_pos)
    
    #Post filtering distances
    filtered_ids = filter_orthologs(ortholog_ids=ortho_ids,
                                        ortholog_type = ortholog_type, 
                                        species_threshold = species_threshold,
                                        drop_duplicates = drop_duplicates ,
                                        #spp_count_proteome_threshold =  spp_count_proteome_threshold,
                                        #ref=False
                                        )
        
    parsed_df = _parse_ids(filtered_ids)
    
    dist_res = {key:dict(v) for key, v in dist_res.items()}
    
    filtered_spp = set(parsed_df['__species'])
    filtered_gpos = set(parsed_df['__gpos'])
    
    filtered_dist_res = {pair:{s:v for s,v in orthologs.items() if s in filtered_spp} for pair,orthologs in dist_res.items() if all(gpos in filtered_gpos for gpos in pair.split('|'))}
    
    #Check number of unique orthologs found overall
    unique_ortho_spp = {spp for pair,orthologs in dist_res.items() for spp in list(orthologs.keys()) if len(str(spp))>3}

    if len(unique_ortho_spp) < n_orthologs:
        print(f'n orthologs in distances less than {n_orthologs}...Removing...')
        #remove family from list
        return 0
    else:
        #Take mean of distances per ortholog
        filtered_dist_res = {pair: {spp:np.mean(dists) for spp, dists in value.items()} for pair, value in filtered_dist_res.items()}
        return filtered_dist_res

# def load_and_filter_homologs(cfg: CommonConfig) -> Optional[pd.DataFrame]:
#     if not cfg.filter_orthologs:
#         return None
#     if cfg.homolog_annotation is None:
#         raise SystemExit("--filter-orthologs requires --homolog-annotation")

#     df = read_table(cfg.homolog_annotation)

#     if cfg.ortholog_types:
#         if "ortholog_type" not in df.columns:
#             raise ValueError("homolog_annotation missing 'ortholog_type' column (edit to match your schema).")
#         df = df[df["ortholog_type"].isin(cfg.ortholog_types)]

#     # TODO: implement species_threshold / proteome_species_count_threshold / min_orthologs using your schema
#     return df

def between_ortholog_divergence(self,
                         gene_pos_list:List[str],
                         ortholog_df:pd.DataFrame,
                         apply_ortholog_filter: bool = False,
                        filter_params: Optional[dict] = None,
                         chunk_size:int=10) -> Dict:
        id_col =self.id_col

        # Ensure the ID column is string-typed and keep the same order as ids_for_matrix
        ortholog_df[id_col] = ortholog_df[id_col].astype(str)
        
        if apply_ortholog_filter: 
            params = filter_params if filter_params is not None else self.ortholog_filter_params

            candidate_ids = ortholog_df[id_col].dropna().unique().tolist()
            filtered_ids = filter_ortholog_region_ids(ortholog_ids = candidate_ids,  **params)

            ortholog_df = ortholog_df[ortholog_df[id_col].isin(filtered_ids)]
            
            if ortholog_df.empty:
                print(f'No orthologs passed filtering criteria:{params}. Gene pos list:{gene_pos_list}')
                return {}                             
        
        df= _parse_ids(ortholog_df[id_col])
        
        # Build group map (species, gpos) -> list of IDs
        grp = df.groupby(['__species', '__gpos'])['__id'].apply(list).to_dict()
        all_species = sorted({sp for sp, _ in grp.keys() if sp != 'HUMAN'})
        gene_pos_set = set(gene_pos_list)
        distances = defaultdict(dict)                        
        
        added_ids,new_dm,updated_ids = self.precompute_ortholog_df(ortholog_df)

        if chunk_size:
            chunks = chunk_list(list(combinations(gene_pos_set,2)),chunk_size)
        else:
            chunks = combinations(gene_pos_set,2)
        
        len_combos =len(chunks)
            
        for i, chunk in enumerate(chunks):
            for j, (gp1, gp2) in enumerate(chunk): 
        #for i, (gp1, gp2) in enumerate(combinations(gene_pos_set,2), start=1):  
                print(f'chunk {i+1}/{len_combos}: {j+1}/{len(chunk)}')
                H1 = grp.get(('HUMAN', gp1), [])
                H2 = grp.get(('HUMAN', gp2), [])
                S1 = [s for sp in all_species for s in grp.get((sp, gp1),[]) if sp != 'HUMAN']
                S2 = [s for sp in all_species for s in grp.get((sp, gp2),[]) if sp != 'HUMAN']
                
                if not (H1 and H2 and S1 and S2): #both need to have at least 1 item
                    continue
                
                # ranks for all pairs
                ranks1 = neighbour_dist_groupwise(
                    group1_ids=H1,
                    group2_ids=S2,
                    added_ids=added_ids,
                    all_ids=list(updated_ids),
                    distance_matrix=new_dm,
                    return_dict=True
                )
                ranks2 = neighbour_dist_groupwise(
                    group1_ids=H2,
                    group2_ids=S1,
                    added_ids=added_ids,
                    all_ids=list(updated_ids),
                    distance_matrix=new_dm,
                    return_dict=True
                )
                
                res1= {ortholog: dist for pair, dist in ranks1.items() for ortholog in all_species if ortholog in pair}
                res2= {ortholog: dist for pair, dist in ranks2.items() for ortholog in all_species if ortholog in pair}

                agg = defaultdict(list)
                
                for ranks in (res1,res2):
                    for spp,dist in ranks.items():
                        agg[spp].append(dist)
                res=agg
                name = f"{gp1}|{gp2}"
                distances[name] = res
        
        return distances  

def within_ortholog_divergence(self,
                                   gene_pos_list:List[str],
                                   ortholog_df:pd.DataFrame,
                                    apply_ortholog_filter: bool = False,
                                    filter_params: Optional[dict] = None,
                                   group_by_gene:bool=False, #LEFT OFF HERE 02-03-26
                            )-> dict: #TODO: type = distancedict class
        """
            Compute within-ortholog divergence between human and other species for each segment(gpos = gene position).

            For each gene position (gpos), compares the neighbour distance of human ortholog region(s)
            to ortholog regions from other species using a precomputed embedding distance matrix.
            Optionally applies ortholog filtering before computation.

            Parameters
            ----------
            gene_pos_list : list[str]
                List of gene position identifiers (gpos) to evaluate.

            ortholog_df : pd.DataFrame
                DataFrame containing ortholog region IDs and metadata. Must include self.id_col.

            apply_ortholog_filter : bool, default=False
                If True, filters ortholog IDs using filter_ortholog_region_ids() before analysis.

            filter_params : dict, optional
                Parameters passed to filter_ortholog_region_ids(). Uses self.ortholog_filter_params if None.

            as_dict : bool, default=False
                If True, returns nested dict: {gpos → {species → distance}}.
                If False, returns aggregated dict: {species → list of distances across gene positions}.

            Returns
            -------
            dict
                Either:

                Nested dict - per gpos (if group_by_gene=True):
                    {gene_position → {species → distance}}

                Aggregated dict - per species (if group_by_gene=False):
                    {species → list of distances}

                Returns empty dict if no valid ortholog comparisons exist.

            Notes
            -----
            Only compares human orthologs against non-human orthologs.
            Requires precomputed embeddings accessible via self.precompute_ortholog_df().
        """
        id_col=self.id_col
        gene_pos_list = list(set(gene_pos_list))

        # Ensure the ID column is string-typed and keep the same order as ids_for_matrix
        ortholog_df[id_col] = ortholog_df[id_col].astype(str)
        
        if apply_ortholog_filter: 
            params = filter_params if filter_params is not None else self.ortholog_filter_params

            candidate_ids = ortholog_df[id_col].dropna().unique().tolist()
            filtered_ids = filter_ortholog_region_ids(ortholog_ids = candidate_ids,  **params)

            ortholog_df = ortholog_df[ortholog_df[id_col].isin(filtered_ids)]
            
            if ortholog_df.empty:
                print(f'No orthologs passed filtering criteria:{params}. Gene pos list:{gene_pos_list}')
                return {} 
        
        df = _parse_ids(ortholog_df[id_col])
        
        # Build group map (species, gpos) -> list of IDs
        grp = df.groupby(['__species', '__gpos'])['__id'].apply(list).to_dict()
        
        all_species = sorted({sp for sp, _ in grp.keys() if sp != 'HUMAN'})
        spp_dist_dicts_ls = []
        spp_dist_dicts = defaultdict(dict)
        
        added_ids,new_dm,updated_ids = self.precompute_ortholog_df(ortholog_df)
        
        for gp in gene_pos_list:
            human_id = grp.get(('HUMAN', gp),[])
            spp_ids = [s for sp in all_species for s in grp.get((sp, gp),[]) if sp != 'HUMAN']
            
            if not human_id or not spp_ids: #both have at least 1 item
                continue
    
            # ranks for all pairs
            ranks = neighbour_dist_groupwise(
                group1_ids=human_id,
                group2_ids=spp_ids,
                added_ids=added_ids,
                all_ids=list(updated_ids),
                distance_matrix=new_dm,
                return_dict=True
            )
            
            res ={}
            for ortholog in all_species:
                for pair, dist in ranks.items():
                    if ortholog in pair:
                        res[ortholog] = dist
            
            if group_by_gene:
                spp_dist_dicts[gp] = res 
            else:
                spp_dist_dicts_ls.append(res)
        
        if group_by_gene:
            return spp_dist_dicts
        else: #aggregate by species
            distances = collect_values_by_key(spp_dist_dicts_ls)
            return distances


def _build_index_maps(ids:list, added_ids:np.ndarray|list|None = None):
    '''Map each ID to its index position'''
    id_to_idx = {id_: k for k, id_ in enumerate(ids)}
    
    if added_ids is not None:
        is_added = np.zeros(len(ids), dtype=bool)
        if isinstance(added_ids, (list, tuple, np.ndarray, set)):
            for a in added_ids:
                if a in id_to_idx:
                    is_added[id_to_idx[a]] = True
    return id_to_idx, is_added #is_added is a mask
    #else:
    #    return id_to_idx

 # def precompute_ortholog_df(self, ortholog_df):
    #     '''
    #     add distance matrix (of ortholog_df to current_df) to current_df
        
    #     :param self: Description
    #     :param ortholog_df: Description
    #     '''
    #     emb_matrix=self.emb_matrix
    #     added_ids = np.array(list(ortholog_df[self.id_col]))
    #     new_dm, updated_ids = emb_matrix.add_to_distance_matrix(ortholog_df)
        
    #     return added_ids,new_dm,updated_ids

# def _load_ortholog_embeddings(ortho_embed_path: str, select_gene_ids: Iterable[str]) -> pd.DataFrame:
#     """Load ortholog embeddings from the specified directory for the given uniprot gene IDs.
#     Args:
#         ortho_dir: Directory containing ortholog embedding files.
#         select_gene_ids: Iterable of uniprot gene IDs to select.
#     Returns:
#         DataFrame containing the concatenated ortholog embeddings."""
#     if os.path.isfile(ortho_embed_path):
#         ortholog_df = _load_embeddings(ortho_embed_path)
#     elif os.path.isdir(ortho_embed_path):
#         ortho_dir = ortho_embed_path
#         #ortholog_df = pd.concat([embedding_h5_to_dataframe(path) for path in find_files_by_identifier(root_path=ortho_dir, select_ids=select_gene_pos, by='gene_pos')])
#         files = find_files_by_prefix(prefixes=select_gene_ids, input_dir =  ortho_dir)
#         #Or find_files_by_identifier
#         if len(files) == 0:
#             raise ValueError(f"No files found in directory {ortho_embed_path} for genes: {select_gene_ids} ")
#         ortholog_df =  pd.concat([_load_embeddings(file_path) for file_path in files])
#     else:
#         raise ValueError(f"ortho_path:{ortho_embed_path} must be an embedding file or directory of embeddings")
#     return ortholog_df

def _load_embeddings(embed_path: str | Path) -> pd.DataFrame:
    """Load the embeddings (csv or .h5)  as DataFrame from path."""
    #embed_path = Path(embed_path)
    if embed_path.endswith('.csv'):
        embed_df = pd.read_csv(embed_path,header=None)
    elif embed_path.endswith('.h5'):
        embed_df = embedding_h5_to_dataframe(embed_path)
    else:
        raise ValueError('Embedding path must be csv or h5')
    return embed_df