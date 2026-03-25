from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Any, Dict, Iterable
import yaml
import pandas as pd
import time
import json
from compute_ndist import ComputeNDistanceDict, _load_embeddings, _load_ortholog_embeddings, _parse_ids,save_pickle, collect_values_by_key, load_fam_map

@dataclass(frozen=True)
class InputsConfig:
    dist_matrix: Optional[Path]
    background_emb: Optional[Path]
    ortholog_emb: Optional[Path]
    homolog_annotation: Optional[Path]
    fam_map: Optional[Path]

@dataclass(frozen=True)
class OutputsConfig:
    out_distance: Path #directory path
    out_random: Path #file path
    overwrite: bool = False

@dataclass(frozen=True)
class ComputeConfig:
    metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"
    chunk_size: int = 5000
    n_jobs: int = 1
    job_id: int = 0

@dataclass(frozen=True)
class NeighbourDivergenceConfig:
    dist_type: Literal["between", "within"]
    segment: str = "IDR"
    return_mode: Literal["aggregate", "per_gene"] = "aggregate"

@dataclass(frozen=True)
class RandomNeighbourDivergenceConfig:
    enabled: bool = False
    dist_type: Literal["between", "within"] = 'within'
    segment: str = "IDR"
    sample_size: int = 100
    random_seed: int = 42
    return_mode: Literal["aggregate", "per_gene"] = "aggregate"

@dataclass(frozen=True)
class OrthologFilterConfig:
    enabled: bool = True
    ortholog_types: Literal["ortholog_one2one","ortholog_one2many","ortholog_one2one_one2many"] = "ortholog_one2one_one2many"
    min_species_coverage: float = 0.5
    min_species: int = 10

@dataclass(frozen=True)
class FNDconfig:
    enabled: bool = False
    fam_distance_matrix: Optional[Path] = None
    bg_distance_matrix: Optional[Path] = None
    output_results: Path = Path("fnd_results.csv")

@dataclass(frozen=True)
class RunConfig:
    verbose: bool = True

@dataclass(frozen=True)
class PipelineConfig:
    inputs: InputsConfig
    outputs: OutputsConfig
    compute: ComputeConfig
    neighbour_divergence: NeighbourDivergenceConfig
    random_neighbour_divergence: RandomNeighbourDivergenceConfig
    ortholog_filter: OrthologFilterConfig
    run: RunConfig
    FND: FNDconfig
    raw: Dict[str, Any]  # keep raw dict for debugging/forward compatibility


def _to_path(x) -> Optional[Path]:
    if x is None:
        return None
    x = str(x).strip()
    return None if x.lower() in {"null", "none", ""} else Path(x)

def load_pipeline_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    # inputs
    inputs = InputsConfig(
        dist_matrix=_to_path(cfg["inputs"].get("dist_matrix")),
        background_emb=_to_path(cfg["inputs"].get("background_emb")),
        ortholog_emb=_to_path(cfg["inputs"].get("ortholog_emb")),
        homolog_annotation=_to_path(cfg["inputs"].get("homolog_annotation")),
        fam_map=_to_path(cfg["inputs"].get("fam_map")),
    )

    outputs = OutputsConfig(
        out_distance=Path(cfg["outputs"]["out_distance"]),
        out_random = Path(cfg["outputs"]["out_random"]),
        overwrite=bool(cfg["outputs"].get("overwrite", False)),
    )

    compute = ComputeConfig(
        metric=cfg["compute"].get("metric", "cosine"),
        chunk_size=int(cfg["compute"].get("chunk_size", 5000)),
        n_jobs=int(cfg["compute"].get("n_jobs", 1)),
        job_id=int(cfg["compute"].get("job_id", 0))
    )

    neighbour_div = NeighbourDivergenceConfig(
        dist_type=cfg["neighbour_divergence"]["dist_type"],
        segment=cfg["neighbour_divergence"].get("segment", "IDR"),
        return_mode=cfg["neighbour_divergence"].get("return_mode", "aggregate"),
    )

    random_neighbour_div = RandomNeighbourDivergenceConfig(
        enabled=bool(cfg["random_neighbour_divergence"].get("enabled", False)),
        dist_type=cfg["random_neighbour_divergence"]["dist_type"],
        segment=cfg["random_neighbour_divergence"].get("segment", "IDR"),
        sample_size=int(cfg["random_neighbour_divergence"].get("sample_size", 100)),
        random_seed=int(cfg["random_neighbour_divergence"].get("random_seed", 42)),
        return_mode=cfg["random_neighbour_divergence"].get("return_mode", "aggregate"),
    )

    ortholog_filter = OrthologFilterConfig(
        enabled=bool(cfg["ortholog_filter"].get("enabled", True)),
        ortholog_types=str(cfg["ortholog_filter"].get("ortholog_types", "ortholog_one2one_one2many")),
        min_species_coverage=float(cfg["ortholog_filter"].get("min_species_coverage", 0.5)),
        min_species=int(cfg["ortholog_filter"].get("min_species", 10)),
    )

    run = RunConfig(verbose=bool(cfg["run"].get("verbose", True)))
    
    FND = FNDconfig( #TODO
        enabled=bool(cfg["calculate_FND"].get("enabled", False)),
        fam_distance_matrix=_to_path(cfg["calculate_FND"].get("fam_distance_matrix")),
        bg_distance_matrix=_to_path(cfg["calculate_FND"].get("bg_distance_matrix")),
        output_results=Path(cfg["calculate_FND"].get("output_results", "fnd_results.csv")),
    ) 

    return PipelineConfig(
        inputs=inputs,
        outputs=outputs,
        compute=compute,
        neighbour_divergence=neighbour_div,
        random_neighbour_divergence=random_neighbour_div,
        ortholog_filter=ortholog_filter,
        run=run,
        FND=FND,
        raw=cfg,
    )


def validate_config(c: PipelineConfig) -> None:
    if c.inputs.dist_matrix is None and c.inputs.background_emb is None:
        raise ValueError("Need either inputs.dist_matrix OR inputs.background_emb.")

    if c.inputs.ortholog_emb is None:
        raise ValueError("inputs.ortholog_emb is required.")

    if c.compute.chunk_size <= 0:
        raise ValueError("compute.chunk_size must be > 0.")

    if c.ortholog_filter.enabled:
        if not (0.0 <= c.ortholog_filter.min_species_coverage <= 1.0):
            raise ValueError("ortholog_filter.min_species_coverage must be between 0 and 1.")
        if c.ortholog_filter.min_species < 1:
            raise ValueError("ortholog_filter.min_species must be >= 1.")
        if c.inputs.homolog_annotation is None:
            raise ValueError("ortholog_filter.enabled=True requires inputs.homolog_annotation.")
    if c.random_neighbour_divergence.enabled: 
        if c.random_neighbour_divergence.sample_size is None:
            raise ValueError("random_neighbour_divergence.sample_size must be set when random_neighbour_divergence.enabled=True.")
        if c.random_neighbour_divergence.sample_size < 1:
            raise ValueError("random_neighbour_divergence.sample_size must be >= 1.")
        #if c.random_neighbour_divergence.sample_size > len(c.inputs.background_emb):
        #    raise ValueError("random_neighbour_divergence.sample_size cannot be greater than the number of available genes in background_emb.")
    


##between_ortholog_divergence, within_ortholog_divergence, save_results


#TODO to implement: parallelization with n_jobs > 1, probably by parallelizing the chunks in between_ortholog_divergence and using multiprocessing or joblib
#Overwrite option, save_results should check if file exists and if overwrite is False, skip computation and loading existing results instead of recomputing
#save_results(results, cfg.outputs.out_results, overwrite=cfg.outputs.overwrite)


def compute_ortholog_divergence(
    engine: ComputeNDistanceDict,
    gene_pos: Iterable[str],
    ortholog_df: pd.DataFrame,
    dist_type: str,
    apply_ortholog_filter: bool,
    filter_params: Optional[Dict[str, Any]] = None,
    return_mode: Literal["aggregate", "per_gene"] = "aggregate",
    chunk_size: Optional[int] = None
):
    """Dispatch to within/between ortholog divergence based on cfg.dist_type."""
    
    common_kwargs = dict(
        gene_pos_list=gene_pos,
        ortholog_df=ortholog_df,
        apply_ortholog_filter=apply_ortholog_filter,
        filter_params=filter_params,
    )

    if dist_type == "between":
        return engine.between_ortholog_divergence(**common_kwargs,
                                                  chunk_size=chunk_size)

    if dist_type == "within":
        return engine.within_ortholog_divergence(
            **common_kwargs,
            group_by_gene=(return_mode == "per_gene"),
        )

    raise ValueError(f"Invalid dist_type: {dist_type}")

##RUN PIPELINES

def run_divergence_pipeline(cfg: PipelineConfig, track=True) -> pd.DataFrame:
    """
    Calculate distance matrix 
    """

    # IO setup
    out_dir = Path(cfg.outputs.out_distance)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if track:
        # Sentinel paths
        success_flag = out_dir / "_SUCCESS.json"
        started_flag = out_dir / "_STARTED.json"
        # Optional: mark started (helpful for debugging)
        started_flag.write_text(json.dumps({
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": f"fam_map={cfg.inputs.fam_map}, ortholog_emb={cfg.inputs.ortholog_emb}, background_emb={cfg.inputs.background_emb}",
        }, indent=2))

    rows: List[Dict[str, Any]] = []
    
    #LOAD ARGUMENTS
    out_dir = cfg.outputs.out_distance
    dist_type = cfg.neighbour_divergence.dist_type
    segment = cfg.neighbour_divergence.segment
    ortholog_emb_path = cfg.inputs.ortholog_emb
    chunk_size = cfg.compute.chunk_size
    
    targets = load_fam_map(cfg.inputs.fam_map)  # Dict[str, List[str]]
    
    apply_ortholog_filter = cfg.ortholog_filter.enabled
    filter_params = None
    if apply_ortholog_filter:
        filter_params = dict(
            relationship=cfg.ortholog_filter.ortholog_types,
            min_species_coverage=cfg.ortholog_filter.min_species_coverage,
            min_species=cfg.ortholog_filter.min_species,
        )
    
     # compute state / distance matrix
    bg_embed_df = _load_embeddings(cfg.inputs.background_emb)
    engine = ComputeNDistanceDict(bg_embed_df)
    
    id_to_idx = {sid: i for i, sid in enumerate(engine.ids)}
    
    n_done = 0
    for (fam_id,genes) in targets.items():
        #fam_id = target.fam_id
        #genes = target.genes
        
        present_ids = [g for g in genes if any(g in id for id in id_to_idx)]
        missing = [g for g in genes if not any(g in id for id in id_to_idx)]

        if not present_ids:
            print(f"No gene IDs found for family {fam_id}, skipping.")
            continue
        if missing:
            print(f"Warning: {len(missing)} gene IDs not found for family {fam_id}: {missing}")
        
        ortho_df = _load_ortholog_embeddings(ortho_embeddings_path=ortholog_emb_path, gene_ids=genes)

        grp_ids = _parse_ids(ortho_df.iloc[:,0])
        gene_pos = set(grp_ids.loc[grp_ids["__genes"].isin(genes), "__gpos"])

        # 2) process
        dist = compute_ortholog_divergence(
            engine=engine,
            gene_pos=gene_pos,
            ortholog_df=ortho_df,
            dist_type=dist_type,
            apply_ortholog_filter=apply_ortholog_filter,
            filter_params= filter_params,
            chunk_size=chunk_size
        )

        # 3) save
        out_path = f"{out_dir}/{fam_id}_{dist_type}_{segment}_diverge"
        save_pickle(dist, out_path)
        print(f"Saved ndist for family {fam_id} to {out_path}")
        n_done += 1
    if track:
        # Only write success *after* loop finishes without exceptions
        success_flag.write_text(json.dumps({
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_written": n_done,
        }, indent=2))

    return pd.DataFrame(rows)

# def shard_targets(targets: dict, job_id: int, num_jobs: int):
#     items = sorted(targets.items(), key=lambda kv: kv[0])  # stable
#     return [kv for i, kv in enumerate(items) if i % num_jobs == job_id]

def shard_targets(targets: dict, job_id: int, num_jobs: int):
    """
    Split targets evenly across jobs.
    Each job receives a contiguous chunk of families.
    """
    items = sorted(targets.items())  # stable ordering
    n = len(items)

    chunk = (n + num_jobs - 1) // num_jobs  # ceiling division
    start = job_id * chunk
    end = min(start + chunk, n)

    return items[start:end]

def run_divergence_pipeline_parallel(cfg: PipelineConfig, track=True) -> pd.DataFrame:
    """
    Calculate distance matrix 
    """

    # IO setup
    out_dir = Path(cfg.outputs.out_distance)
    out_dir.mkdir(parents=True, exist_ok=True)

    #TODO add to config
    num_jobs = getattr(cfg.compute, "n_jobs", 1)
    job_id = getattr(cfg.compute, "job_id", 0)
    job_id = int(job_id)
    num_jobs=int(num_jobs)
    print('Job id',job_id)
    print('num_jobs', num_jobs)
    if num_jobs < 1:
        raise ValueError("cfg.compute.num_jobs musqt be >= 1")
    if not (0 <= job_id < num_jobs):
        raise ValueError("cfg.compute.job_id must be in [0, num_jobs)")
    
    if track:
        # Sentinel paths
        success_flag = out_dir / "_SUCCESS.json"
        started_flag = out_dir / "_STARTED.json"
        # Optional: mark started (helpful for debugging)
        started_flag.write_text(json.dumps({
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": f"fam_map={cfg.inputs.fam_map}, ortholog_emb={cfg.inputs.ortholog_emb}, background_emb={cfg.inputs.background_emb}",
        }, indent=2))

    rows: List[Dict[str, Any]] = []
    
    #LOAD ARGUMENTS
    out_dir = cfg.outputs.out_distance
    dist_type = cfg.neighbour_divergence.dist_type
    segment = cfg.neighbour_divergence.segment
    ortholog_emb_path = cfg.inputs.ortholog_emb
    chunk_size = cfg.compute.chunk_size
    
    targets = load_fam_map(cfg.inputs.fam_map)  # Dict[str, List[str]]
    target_items = shard_targets(targets, job_id=job_id, num_jobs=num_jobs)
    
    apply_ortholog_filter = cfg.ortholog_filter.enabled
    filter_params = None
    if apply_ortholog_filter:
        filter_params = dict(
            relationship=cfg.ortholog_filter.ortholog_types,
            min_species_coverage=cfg.ortholog_filter.min_species_coverage,
            min_species=cfg.ortholog_filter.min_species,
        )
    
     # compute state / distance matrix
    bg_embed_df = _load_embeddings(cfg.inputs.background_emb)
    engine = ComputeNDistanceDict(bg_embed_df)
    
    id_to_idx = {sid: i for i, sid in enumerate(engine.ids)}
    
    n_done = 0
    for (fam_id,genes) in target_items:
        #fam_id = target.fam_id
        #genes = target.genes
        
        present_ids = [g for g in genes if any(g in id for id in id_to_idx)]
        missing = [g for g in genes if not any(g in id for id in id_to_idx)]

        if not present_ids:
            print(f"[job {job_id}/{num_jobs}]: No gene IDs found for family {fam_id}, skipping.")
            continue
        if missing:
            print(f"[job {job_id}/{num_jobs}] Warning: {len(missing)} gene IDs not found for family {fam_id}: {missing}")
        
        ortho_df = _load_ortholog_embeddings(ortho_embeddings_path=ortholog_emb_path, gene_ids=genes)

        grp_ids = _parse_ids(ortho_df.iloc[:,0])
        gene_pos = set(grp_ids.loc[grp_ids["__genes"].isin(genes), "__gpos"])

        # 2) process
        dist = compute_ortholog_divergence(
            engine=engine,
            gene_pos=gene_pos,
            ortholog_df=ortho_df,
            dist_type=dist_type,
            apply_ortholog_filter=apply_ortholog_filter,
            filter_params= filter_params,
            chunk_size=chunk_size
        )

        # 3) save
        out_path = f"{out_dir}/{fam_id}_{dist_type}_{segment}_diverge"
        save_pickle(dist, out_path)
        #print(f"Saved ndist for family {fam_id} to {out_path}")
        print(f"[job {job_id}/{num_jobs}] Saved ndist for family {fam_id} to {out_path}")
        n_done += 1
    if track:
        # Only write success *after* loop finishes without exceptions
        success_flag.write_text(json.dumps({
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": job_id,
            "num_jobs": num_jobs,
            "n_written": n_done,
        }, indent=2))
        
    print(f"Families assigned to this job:")
    for fam_id, _ in target_items:
        print("  ", fam_id)

    return pd.DataFrame(rows)

def run_random_divergence_pipeline(cfg:PipelineConfig
) -> pd.DataFrame:
    """
    Calculate distance matrix 
    """
    
    out_dir = cfg.outputs.out_distance
    out_path = cfg.outputs.out_random
    
    dist_type = cfg.random_neighbour_divergence.dist_type
    segment = cfg.random_neighbour_divergence.segment
    sample_size = cfg.random_neighbour_divergence.sample_size
    random_seed = cfg.random_neighbour_divergence.random_seed

    #filtering
    apply_ortholog_filter = cfg.ortholog_filter.enabled
    filter_params = None
    if apply_ortholog_filter:
        filter_params = dict(
            relationship=cfg.ortholog_filter.ortholog_types,
            min_species_coverage=cfg.ortholog_filter.min_species_coverage,
            min_species=cfg.ortholog_filter.min_species,
        )
    
    bg_embed_df = _load_embeddings(cfg.inputs.background_emb)
    engine = ComputeNDistanceDict(bg_embed_df)
    
    # Sample genes uniformly
    bg_ids = _parse_ids(engine.ids)
    genes = bg_ids["__genes"].dropna().unique()
    if sample_size > len(genes):
        raise ValueError(f"sample_size={sample_size} > available genes={len(genes)}")

    rand_genes = (pd.Series(genes).sample(n=sample_size, random_state=random_seed).tolist())
    rand_gpos = (bg_ids.loc[bg_ids["__genes"].isin(rand_genes), "__gpos"].dropna().unique().tolist())

    #load random embeddings
    print('load ortho embeddings...')
    ortho_df= _load_ortholog_embeddings(
                ortho_embeddings_path=cfg.inputs.ortholog_emb,
                gene_ids=rand_genes
            )
    
    ortho_ids_df = _parse_ids(ortho_df)
    
    
    
    # 2) process
    print('compute divergence')
    dist = compute_ortholog_divergence(
                engine=engine,
                gene_pos=rand_gpos,
                ortholog_df=ortho_df,
                dist_type=dist_type,
                apply_ortholog_filter=apply_ortholog_filter,
                filter_params=filter_params
            )
    #REPORT HOW MANY SPECIES/GENES LOST IN FILTERING, probably have to oversample? Or don't filter the random bg?
    
    # Aggregate
    #rand_agg = collect_values_by_key(list(dist.values()))

    # Save
    if out_path is None:
        out_path = Path(out_dir) / f"random_{sample_size}_{dist_type}_{segment}_diverge"
    save_pickle(dist, str(out_path))
    print(f"Saved to {out_path}.pkl")
    
    return #out_path #TODO


import time
from pathlib import Path

def _wait_for_success_flag(out_dir: Path, timeout_s: int = 86400, poll_s: int = 10) -> None:
    flag = out_dir / "_SUCCESS.json"
    t0 = time.time()
    while not flag.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for success flag: {flag}")
        time.sleep(poll_s)


def _wait_for_file(path: Path, timeout_s: int = 86400, poll_s: int = 10) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(poll_s)


from Paper.src.distances.compute_ndist import calc_FND

from pathlib import Path
import pandas as pd

def run_FND_pipeline(cfg: "PipelineConfig") -> pd.DataFrame:
    """
    Compute Family Neighbour Divergence (FND), ensuring prerequisite distance outputs exist.

    Inputs
    ------
    cfg.FND.fam_distance_matrix : Optional[Union[str, Path]]
        Family-distance input. May be:
        - a FILE path (single aggregated family distance object), or
        - a DIRECTORY containing per-family distance outputs (e.g. '*diverge.pkl') and a
          sentinel success flag (e.g. '_SUCCESS.json') written when the divergence pipeline completes.

        If None, this function runs `run_divergence_pipeline(cfg)` and uses `cfg.outputs.out_distance`
        as the family-distance directory.

    cfg.FND.bg_distance_matrix : Optional[Union[str, Path]]
        Background/random distance input. Must be a FILE path.
        If None, this function runs `run_random_divergence_pipeline(cfg)` and uses a default output
        file path (recommended to store in config).

    Behavior
    --------
    - If fam_distance_matrix is a directory: waits for a success flag in that directory.
      If it is a file: waits for the file to exist.
    - Always waits for the background distance file to exist.
    - Calls `calc_FND(...)` exactly once.

    Returns
    -------
    pd.DataFrame
        FND results returned by `calc_FND`.
    """
    # Resolve family distance path (dir or file)
    fam_dist_path = cfg.FND.fam_distance_matrix
    if fam_dist_path is None:
        run_divergence_pipeline(cfg)
        fam_dist_path = Path(cfg.outputs.out_distance)   # directory expected to contain _SUCCESS.json
    else:
        fam_dist_path = Path(fam_dist_path)

    # Resolve background distance path (file)
    bg_dist_path = cfg.FND.bg_distance_matrix
    if bg_dist_path is None:
        # Configure + run background/random pipeline
        cfg.random_neighbour_divergence.enabled = True
        cfg.random_neighbour_divergence.sample_size = 100
        cfg.random_neighbour_divergence.dist_type = "within"
        
        # put this in config as cfg.FND.bg_distance_matrix_output or similar
        bg_dist_path = Path(cfg.outputs.out_distance) / "random_bg_distance_matrix.pkl"
        
        cfg.output.out_random =  bg_dist_path

        run_random_divergence_pipeline(cfg)

        
    else:
        bg_dist_path = Path(bg_dist_path)

    # Wait for prerequisites
    if fam_dist_path.is_dir():
        _wait_for_success_flag(fam_dist_path)   # waits for fam_dist_path/_SUCCESS.json (or whatever you implement)
    else:
        _wait_for_file(fam_dist_path)

    _wait_for_file(bg_dist_path)

    # Compute FND
    return calc_FND(
        fam_dist_path,
        bg_dist_path,
        data_transform="log_mean",
        save_path=cfg.FND.output_results,
    )

    

def run_pipeline(cfg_path: str | Path):
    cfg = load_pipeline_config(cfg_path)
    validate_config(cfg)
    
    print(cfg)
    
    if cfg.FND.enabled:
        
        # For now, just compute FND directly from distance matrix if provided, otherwise compute distance matrix from embeddings and then compute FND, but in the future could add option to compute FND directly from embeddings without computing full distance matrix by using a more efficient algorithm that only computes distances to nearest neighbours instead of all pairwise distances
        run_FND_pipeline(cfg)
        return
    
    #if random divergence, call that pipeline instead
    if cfg.random_neighbour_divergence.enabled:
        print('Running random neighbour divergence...')
        run_random_divergence_pipeline(cfg)
        return
    
    if cfg.compute.n_jobs > 1:
        print(f'Running family neighbour divergence in parallel ({cfg.compute.n_jobs} jobs)...')
        #print("Warning: n_jobs > 1 is not currently implemented, running with n_jobs=1.") #TODO: implement parallelization
        run_divergence_pipeline_parallel(cfg)
    
    else:
        print('Running family neighbour divergence...')
        # for now just run neighbour divergence, can add CLI args to specify which pipeline to run if needed
        run_divergence_pipeline(cfg)
        return
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ortholog divergence pipeline.")
    parser.add_argument("config", type=str, help="Path to YAML config file.")
    args = parser.parse_args()

    run_pipeline(args.config)
    

#TODO
#incoporate FND arguments into pipeline config args
#incorporate option to compute FND directly from distance matrices if provided, otherwise compute distance matrices from embeddings and then compute FND, into run_FND_pipeline function
#



