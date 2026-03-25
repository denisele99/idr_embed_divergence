import shlex, json
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

def config_to_argv(config_path: Path) -> list[str]:
    suf = config_path.suffix.lower()
    if suf == ".json":
        cfg = json.loads(config_path.read_text())
    elif suf in [".yml", ".yaml"]:
        if yaml is None:
            raise SystemExit("PyYAML not installed. pip install pyyaml")
        cfg = yaml.safe_load(config_path.read_text()) or {}
    else:
        raise SystemExit("Config must be .json or .yml/.yaml")

    subcmd = cfg.get("run", {}).get("subcommand")
    if subcmd not in {"genes", "fam-map", "random"}:
        raise SystemExit("config.run.subcommand must be genes|fam-map|random")

    common = cfg.get("common", {})
    targets = cfg.get("targets", {})

    argv = ["run", subcmd]

    # common flags (only add if present / true)
    def add_flag(k, flag, transform=str):
        v = common.get(k, None)
        if v is None:
            return
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            return
        argv.extend([flag, transform(v)])

    add_flag("dist_matrix", "--dist-matrix")
    add_flag("background_emb", "--background-emb")
    add_flag("ortholog_emb", "--ortholog-emb")
    add_flag("out_results", "--out-results")
    add_flag("out_matrix", "--out-matrix")
    add_flag("metric", "--metric")
    add_flag("dtype", "--dtype")
    add_flag("chunk_size", "--chunk-size", int)
    add_flag("n_jobs", "--n-jobs", int)
    add_flag("id_col", "--id-col", int)
    add_flag("embedding_cols", "--embedding-cols")

    # booleans
    if common.get("overwrite"):
        argv.append("--overwrite")
    if "normalize" in common and common["normalize"] is False:
        argv.append("--no-normalize")
    elif common.get("normalize") is True:
        argv.append("--normalize")
    if common.get("filter_orthologs"):
        argv.append("--filter-orthologs")
    if common.get("verbose"):
        argv.append("--verbose")

    # filtering
    add_flag("homolog_annotation", "--homolog-annotation")
    for t in (common.get("ortholog_type") or []):
        argv.extend(["--ortholog-type", str(t)])
    add_flag("species_threshold", "--species-threshold", float)
    add_flag("proteome_species_count_threshold", "--proteome-species-count-threshold", int)
    add_flag("min_orthologs", "--min-orthologs", int)

    # targets by subcmd
    if subcmd == "genes":
        genes = targets.get("gene_ids") or []
        if genes:
            argv.extend(["--gene-ids", ",".join(map(str, genes))])
        if targets.get("one_family"):
            argv.append("--one-family")

    elif subcmd == "fam-map":
        fam_map = targets.get("fam_map")
        if fam_map:
            argv.extend(["--fam-map", str(fam_map)])
        only = targets.get("only_fams")
        if only:
            argv.extend(["--only-fams", ",".join(map(str, only))])

    elif subcmd == "random":
        for key, flag in [
            ("universe_file", "--universe-file"),
            ("exclude_file", "--exclude-file"),
            ("n_fams", "--n-fams"),
            ("fam_size", "--fam-size"),
            ("seed", "--seed"),
        ]:
            v = targets.get(key)
            if v is not None:
                argv.extend([flag, str(v)])
        if targets.get("allow_overlap"):
            argv.append("--allow-overlap")

    return argv


#!/usr/bin/env python3
"""
nn_divergence_cli_level1_with_config.py

Level 1 (subcommands) + config file support (YAML/JSON) + CLI overrides.

You can run:
  python nn_divergence_cli_level1_with_config.py --config config.yml
or:
  python nn_divergence_cli.py run genes --dist-matrix matrix.npz --gene-ids CDK1,CDK2 --out-results out.csv




Config supports:
  run.subcommand: genes | fam-map | random
  common: (shared flags)
  targets: (subcommand-specific)

See example configs at the bottom of this file (in comments).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class MatrixWithIds:
    D: np.ndarray
    ids: List[str]


@dataclass(frozen=True)
class FamilyTarget:
    fam_id: str
    genes: List[str]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class CommonConfig:
    dist_matrix: Optional[Path]
    background_emb: Optional[Path]
    ortholog_emb: Optional[Path]

    out_results: Path
    out_matrix: Optional[Path]
    overwrite: bool

    metric: str
    normalize: bool
    dtype: str
    chunk_size: int
    n_jobs: int

    id_col: int
    embedding_cols: Optional[str]

    filter_orthologs: bool
    homolog_annotation: Optional[Path]
    ortholog_types: Optional[List[str]]
    species_threshold: float
    proteome_species_count_threshold: int
    min_orthologs: int

    verbose: bool




# ----------------------------
# IO helpers
# ----------------------------

def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def parse_embedding_cols(df: pd.DataFrame, id_col: int, embedding_cols: Optional[str]) -> Tuple[pd.Series, np.ndarray]:
    if id_col < 0 or id_col >= df.shape[1]:
        raise ValueError(f"id_col={id_col} out of bounds for df with {df.shape[1]} columns.")
    ids = df.iloc[:, id_col].astype(str)

    if embedding_cols is None:
        emb_df = df.drop(df.columns[id_col], axis=1)
        return ids, emb_df.to_numpy()

    spec = embedding_cols.strip()
    if ":" in spec:
        a, b = spec.split(":", 1)
        start = int(a) if a != "" else None
        end = int(b) if b != "" else None
        return ids, df.iloc[:, slice(start, end)].to_numpy()

    cols = [int(x.strip()) for x in spec.split(",") if x.strip()]
    return ids, df.iloc[:, cols].to_numpy()


def load_embeddings(path_or_dir: Path, id_col: int, embedding_cols: Optional[str]) -> Tuple[List[str], np.ndarray]:
    supported = {".csv", ".tsv", ".txt", ".parquet"}
    if path_or_dir.is_dir():
        files = sorted([p for p in path_or_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported])
        if not files:
            raise ValueError(f"No supported embedding files found in directory: {path_or_dir}")
        all_ids: List[str] = []
        all_X: List[np.ndarray] = []
        for f in files:
            df = read_table(f)
            ids, X = parse_embedding_cols(df, id_col=id_col, embedding_cols=embedding_cols)
            all_ids.extend(ids.tolist())
            all_X.append(X)
        return all_ids, np.vstack(all_X)

    df = read_table(path_or_dir)
    ids, X = parse_embedding_cols(df, id_col=id_col, embedding_cols=embedding_cols)
    return ids.tolist(), X


def save_results_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".csv":
        df.to_csv(path, index=False)
        return
    if suf in [".tsv", ".txt"]:
        df.to_csv(path, index=False, sep="\t")
        return
    raise ValueError(f"Unsupported output format for results: {path}")


def save_matrix(matrix: MatrixWithIds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".npz":
        np.savez_compressed(path, D=matrix.D, ids=np.array(matrix.ids, dtype=object))
        return
    if suf in [".csv", ".tsv", ".txt"]:
        sep = "," if suf == ".csv" else "\t"
        pd.DataFrame(matrix.D, index=matrix.ids, columns=matrix.ids).to_csv(path, sep=sep)
        return
    raise ValueError(f"Unsupported matrix output format: {path} (use .npz or labeled .csv/.tsv)")


def load_matrix(path: Path) -> MatrixWithIds:
    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        return MatrixWithIds(D=z["D"], ids=z["ids"].astype(str).tolist())
    if suf in [".csv", ".tsv", ".txt"]:
        sep = "," if suf == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        return MatrixWithIds(D=df.to_numpy(), ids=df.index.astype(str).tolist())
    raise ValueError(f"Unsupported matrix format: {path}")


# ----------------------------
# Distance computation (dense scaffold)
# ----------------------------

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def compute_distance_matrix_dense(
    ids: List[str],
    X: np.ndarray,
    metric: str,
    normalize: bool,
    dtype: str,
    chunk_size: int,
    verbose: bool,
) -> MatrixWithIds:
    X = X.astype(dtype, copy=False)
    if normalize and metric == "cosine":
        X = l2_normalize(X)

    n = X.shape[0]
    D = np.empty((n, n), dtype=dtype)

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        Xi = X[i0:i1]

        if metric == "cosine":
            sim = Xi @ X.T
            Di = 1.0 - sim
        elif metric == "euclidean":
            a2 = np.sum(Xi * Xi, axis=1, keepdims=True)
            b2 = np.sum(X * X, axis=1, keepdims=True).T
            Di = np.sqrt(np.maximum(a2 + b2 - 2.0 * (Xi @ X.T), 0.0))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        D[i0:i1, :] = Di.astype(dtype, copy=False)
        if verbose:
            print(f"[matrix] rows {i0}:{i1}/{n}", file=sys.stderr)

    return MatrixWithIds(D=D, ids=ids)


# ----------------------------
# Filtering scaffold (adapt to your schema)
# ----------------------------

def load_and_filter_homologs(cfg: CommonConfig) -> Optional[pd.DataFrame]:
    if not cfg.filter_orthologs:
        return None
    if cfg.homolog_annotation is None:
        raise SystemExit("--filter-orthologs requires --homolog-annotation")

    df = read_table(cfg.homolog_annotation)

    if cfg.ortholog_types:
        if "ortholog_type" not in df.columns:
            raise ValueError("homolog_annotation missing 'ortholog_type' column (edit to match your schema).")
        df = df[df["ortholog_type"].isin(cfg.ortholog_types)]

    # TODO: implement species_threshold / proteome_species_count_threshold / min_orthologs using your schema
    return df


# ----------------------------
# Target builders
# ----------------------------

def parse_gene_list(gene_ids: Optional[str], gene_id: Optional[List[str]], gene_id_file: Optional[Path]) -> List[str]:
    out: List[str] = []
    if gene_ids:
        out.extend([x.strip() for x in gene_ids.split(",") if x.strip()])
    if gene_id:
        out.extend([x.strip() for x in gene_id if x.strip()])
    if gene_id_file:
        lines = gene_id_file.read_text().splitlines()
        out.extend([ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")])

    seen = set()
    dedup: List[str] = []
    for g in out:
        if g not in seen:
            seen.add(g)
            dedup.append(g)
    return dedup


def targets_from_genes(genes: List[str], one_family: bool) -> List[FamilyTarget]:
    if not genes:
        raise SystemExit("No genes provided.")
    if one_family:
        return [FamilyTarget(fam_id="GENE_SET", genes=genes, meta={"mode": "genes", "one_family": True})]
    return [FamilyTarget(fam_id=g, genes=[g], meta={"mode": "genes", "one_family": False})]


def load_fam_map(path: Path) -> Dict[str, List[str]]:
    suf = path.suffix.lower()

    if suf == ".json":
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("fam-map JSON must be a dict: {fam_id: [genes...]}")
        return {str(k): [str(x) for x in v] for k, v in obj.items()}

    if suf in [".yml", ".yaml"]:
        if yaml is None:
            raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
        obj = yaml.safe_load(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("fam-map YAML must be a dict: {fam_id: [genes...]}")
        return {str(k): [str(x) for x in v] for k, v in obj.items()}

    if suf in [".tsv", ".txt", ".csv"]:
        sep = "," if suf == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        required = {"fam_id", "gene"}
        if not required.issubset(df.columns):
            raise ValueError(f"fam-map table must have columns {required}, found: {list(df.columns)}")
        fam_to_genes: Dict[str, List[str]] = {}
        for fam, sub in df.groupby("fam_id"):
            fam_to_genes[str(fam)] = sub["gene"].astype(str).tolist()
        return fam_to_genes

    raise ValueError("Unsupported fam-map format. Use .json/.yml/.tsv/.csv")


def targets_from_fam_map(fam_to_genes: Dict[str, List[str]]) -> List[FamilyTarget]:
    targets: List[FamilyTarget] = []
    for fam_id, genes in fam_to_genes.items():
        genes = [g for g in genes if str(g).strip()]
        if genes:
            targets.append(FamilyTarget(fam_id=str(fam_id), genes=genes, meta={"mode": "fam-map"}))
    if not targets:
        raise SystemExit("Family map produced 0 targets.")
    return targets


def targets_random(universe: List[str], n_fams: int, fam_size: int, seed: int, allow_overlap: bool) -> List[FamilyTarget]:
    if n_fams <= 0 or fam_size <= 0:
        raise SystemExit("--n-fams and --fam-size must be > 0")
    universe = list(dict.fromkeys(universe))
    if len(universe) < fam_size:
        raise SystemExit(f"Universe has {len(universe)} genes, but fam_size={fam_size} requested.")

    rng = np.random.default_rng(seed)
    used: set[str] = set()
    targets: List[FamilyTarget] = []

    for i in range(n_fams):
        if allow_overlap:
            picks = rng.choice(universe, size=fam_size, replace=False).tolist()
        else:
            remaining = [g for g in universe if g not in used]
            if len(remaining) < fam_size:
                raise SystemExit("Not enough remaining genes to sample without overlap. Use allow_overlap or reduce sizes.")
            picks = rng.choice(remaining, size=fam_size, replace=False).tolist()
            used.update(picks)

        targets.append(FamilyTarget(fam_id=f"RANDOM_{i:04d}", genes=picks, meta={"mode": "random", "seed": seed}))
    return targets


# ----------------------------
# Core divergence computation (stub)
# ----------------------------

# def compute_divergence(
#     matrix: MatrixWithIds,
#     targets: List[FamilyTarget],
#     homolog_df: Optional[pd.DataFrame],
#     verbose: bool,
# ) -> pd.DataFrame:
#     """
#     Replace with your real divergence calculation.
#     """
#     id_to_idx = {sid: i for i, sid in enumerate(matrix.ids)}
#     rows: List[Dict[str, Any]] = []

#     for t in targets:
#         present = [g for g in t.genes if g in id_to_idx]
#         missing = [g for g in t.genes if g not in id_to_idx]

#         mean_within = float("nan")
#         if len(present) >= 2:
#             idxs = [id_to_idx[g] for g in present]
#             subD = matrix.D[np.ix_(idxs, idxs)]
#             mean_within = float(np.mean(subD))

#         rows.append(
#             {
#                 "fam_id": t.fam_id,
#                 "n_genes_input": len(t.genes),
#                 "n_present": len(present),
#                 "n_missing": len(missing),
#                 "missing_genes": ";".join(missing) if missing else "",
#                 "mean_within_target_distance": mean_within,
#                 "homolog_rows_used": int(len(homolog_df)) if homolog_df is not None else 0,
#             }
#         )
#         if verbose:
#             print(f"[analysis] {t.fam_id}: present={len(present)} missing={len(missing)}", file=sys.stderr)

#     return pd.DataFrame(rows)


# ----------------------------
# Shared pipeline runner (Level 1)
# ----------------------------

def build_common_config(args: argparse.Namespace) -> CommonConfig:
    return CommonConfig(
        dist_matrix=args.dist_matrix,
        background_emb=args.background_emb,
        ortholog_emb=args.ortholog_emb,
        out_results=args.out_results,
        out_matrix=args.out_matrix,
        overwrite=args.overwrite,
        metric=args.metric,
        normalize=args.normalize,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        id_col=args.id_col,
        embedding_cols=args.embedding_cols,
        filter_orthologs=args.filter_orthologs,
        homolog_annotation=args.homolog_annotation,
        ortholog_types=args.ortholog_type if args.ortholog_type else None,
        species_threshold=args.species_threshold,
        proteome_species_count_threshold=args.proteome_species_count_threshold,
        min_orthologs=args.min_orthologs,
        verbose=args.verbose,
    )


def validate_common(cfg: CommonConfig) -> None:
    if cfg.dist_matrix is None:
        if cfg.background_emb is None or cfg.ortholog_emb is None:
            raise SystemExit("Provide --dist-matrix OR (--background-emb AND --ortholog-emb).")

    for outp in [cfg.out_results, cfg.out_matrix]:
        if outp and outp.exists() and not cfg.overwrite:
            raise SystemExit(f"Output exists: {outp}. Use --overwrite to replace it.")

    if cfg.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0.")


def get_or_build_matrix(cfg: CommonConfig) -> MatrixWithIds:
    if cfg.dist_matrix is not None:
        if cfg.verbose:
            print(f"[io] loading matrix: {cfg.dist_matrix}", file=sys.stderr)
        return load_matrix(cfg.dist_matrix)

    assert cfg.background_emb is not None and cfg.ortholog_emb is not None

    if cfg.verbose:
        print(f"[io] loading background embeddings: {cfg.background_emb}", file=sys.stderr)
    bg_ids, bg_X = load_embeddings(cfg.background_emb, id_col=cfg.id_col, embedding_cols=cfg.embedding_cols)

    if cfg.verbose:
        print(f"[io] loading ortholog embeddings: {cfg.ortholog_emb}", file=sys.stderr)
    ortho_ids, ortho_X = load_embeddings(cfg.ortholog_emb, id_col=cfg.id_col, embedding_cols=cfg.embedding_cols)

    all_ids = bg_ids + ortho_ids
    all_X = np.vstack([bg_X, ortho_X])

    if cfg.verbose:
        print(f"[matrix] computing dense {len(all_ids)}x{len(all_ids)} distances (metric={cfg.metric})", file=sys.stderr)

    return compute_distance_matrix_dense(
        ids=all_ids,
        X=all_X,
        metric=cfg.metric,
        normalize=cfg.normalize,
        dtype=cfg.dtype,
        chunk_size=cfg.chunk_size,
        verbose=cfg.verbose,
    )


def run_pipeline(args: argparse.Namespace, targets: List[FamilyTarget]) -> None:
    cfg = build_common_config(args)
    validate_common(cfg)

    homolog_df = load_and_filter_homologs(cfg)
    matrix = get_or_build_matrix(cfg)

    if cfg.out_matrix is not None:
        if cfg.verbose:
            print(f"[io] saving matrix: {cfg.out_matrix}", file=sys.stderr)
        save_matrix(matrix, cfg.out_matrix)

    results = compute_divergence(matrix, targets, homolog_df, verbose=cfg.verbose)

    if cfg.verbose:
        print(f"[io] saving results: {cfg.out_results}", file=sys.stderr)
    save_results_table(results, cfg.out_results)

    if cfg.verbose:
        print("[done]", file=sys.stderr)


# ----------------------------
# CLI: subcommands + target builder registry
# ----------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    mx = p.add_mutually_exclusive_group(required=True)
    mx.add_argument("--dist-matrix", type=Path, default=None, help="Precomputed matrix (.npz recommended).")

    # embeddings mode (validated manually)
    p.add_argument("--background-emb", type=Path, default=None, help="Background embeddings file.")
    p.add_argument("--ortholog-emb", type=Path, default=None, help="Ortholog embeddings file or directory.")

    # outputs
    p.add_argument("--out-results", type=Path, required=True)
    p.add_argument("--out-matrix", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")

    # distance settings
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--n-jobs", type=int, default=1)

    # parsing
    p.add_argument("--id-col", type=int, default=0)
    p.add_argument("--embedding-cols", type=str, default=None)

    # filtering
    p.add_argument("--filter-orthologs", action="store_true")
    p.add_argument("--homolog-annotation", type=Path, default=None)
    p.add_argument("--ortholog-type", action="append", default=None)
    p.add_argument("--species-threshold", type=float, default=0.5)
    p.add_argument("--proteome-species-count-threshold", type=int, default=1)
    p.add_argument("--min-orthologs", type=int, default=10)

    p.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="nn-divergence",
        description="Neighbour divergence CLI (Level 1) + --config YAML/JSON expansion.",
    )
    # NOTE: --config is handled by a pre-parser, but we include it here so it shows up in --help.
    root.add_argument("--config", type=Path, default=None, help="YAML/JSON config file (expanded into CLI args).")

    sub = root.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run pipeline")
    run_sub = run.add_subparsers(dest="targets_cmd", required=True)

    # genes
    p_genes = run_sub.add_parser("genes", help="Targets from gene list")
    add_common_args(p_genes)
    p_genes.add_argument("--gene-ids", type=str, default=None, help="Comma-separated gene IDs.")
    p_genes.add_argument("--gene-id", action="append", default=None, help="Repeatable gene ID.")
    p_genes.add_argument("--gene-id-file", type=Path, default=None, help="One gene ID per line.")
    p_genes.add_argument("--one-family", action="store_true",
                         help="Treat all provided genes as one target family (GENE_SET). Default: one family per gene.")

    # fam-map
    p_map = run_sub.add_parser("fam-map", help="Targets from family map file")
    add_common_args(p_map)
    p_map.add_argument("--fam-map", type=Path, required=True, help="Family map: .json/.yml or table with fam_id,gene.")
    p_map.add_argument("--only-fams", type=str, default=None, help="Comma-separated fam_ids to keep (optional).")

    # random
    p_rand = run_sub.add_parser("random", help="Random targets from a universe")
    add_common_args(p_rand)
    p_rand.add_argument("--universe-file", type=Path, required=True, help="One gene ID per line.")
    p_rand.add_argument("--exclude-file", type=Path, default=None, help="Optional file of genes to exclude.")
    p_rand.add_argument("--n-fams", type=int, required=True)
    p_rand.add_argument("--fam-size", type=int, required=True)
    p_rand.add_argument("--seed", type=int, default=1)
    p_rand.add_argument("--allow-overlap", action="store_true")

    return root


def build_targets_genes(args: argparse.Namespace) -> List[FamilyTarget]:
    genes = parse_gene_list(args.gene_ids, args.gene_id, args.gene_id_file)
    return targets_from_genes(genes, one_family=args.one_family)


def build_targets_fam_map(args: argparse.Namespace) -> List[FamilyTarget]:
    fam_to_genes = load_fam_map(args.fam_map)
    if args.only_fams:
        keep = {x.strip() for x in args.only_fams.split(",") if x.strip()}
        fam_to_genes = {k: v for k, v in fam_to_genes.items() if k in keep}
    return targets_from_fam_map(fam_to_genes)


def build_targets_random(args: argparse.Namespace) -> List[FamilyTarget]:
    universe = [
        ln.strip()
        for ln in args.universe_file.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.exclude_file:
        exclude = {
            ln.strip()
            for ln in args.exclude_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
        universe = [g for g in universe if g not in exclude]
    return targets_random(universe, args.n_fams, args.fam_size, args.seed, args.allow_overlap)


TARGET_BUILDERS: Dict[str, Callable[[argparse.Namespace], List[FamilyTarget]]] = {
    "genes": build_targets_genes,
    "fam-map": build_targets_fam_map,
    "random": build_targets_random,
}


# ----------------------------
# Config expansion (YAML/JSON -> argv) + overrides
# ----------------------------

# Add these imports near the top
import json
from pathlib import Path

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


def load_cfg(path: Path) -> dict:
    suf = path.suffix.lower()
    if suf == ".json":
        return json.loads(path.read_text()) or {}
    if suf in [".yml", ".yaml"]:
        if yaml is None:
            raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
        return yaml.safe_load(path.read_text()) or {}
    raise SystemExit("Config must be .json or .yml/.yaml")


def cfg_to_argv(cfg: dict) -> list[str]:
    subcmd = (cfg.get("run") or {}).get("subcommand")
    if subcmd not in {"genes", "fam-map", "random"}:
        raise SystemExit("Config requires run.subcommand in {genes, fam-map, random}")

    common = cfg.get("common") or {}
    targets = cfg.get("targets") or {}

    argv = ["run", subcmd]

    def add_opt(flag, val):
        argv.extend([flag, str(val)])

    def add_bool(flag, val):
        if bool(val):
            argv.append(flag)

    # common -> flags
    for k, flag in [
        ("dist_matrix", "--dist-matrix"),
        ("background_emb", "--background-emb"),
        ("ortholog_emb", "--ortholog-emb"),
        ("out_results", "--out-results"),
        ("out_matrix", "--out-matrix"),
        ("metric", "--metric"),
        ("dtype", "--dtype"),
        ("chunk_size", "--chunk-size"),
        ("n_jobs", "--n-jobs"),
        ("id_col", "--id-col"),
        ("embedding_cols", "--embedding-cols"),
        ("homolog_annotation", "--homolog-annotation"),
        ("species_threshold", "--species-threshold"),
        ("proteome_species_count_threshold", "--proteome-species-count-threshold"),
        ("min_orthologs", "--min-orthologs"),
    ]:
        if common.get(k) is not None:
            add_opt(flag, common[k])

    add_bool("--overwrite", common.get("overwrite", False))
    add_bool("--filter-orthologs", common.get("filter_orthologs", False))
    add_bool("--verbose", common.get("verbose", False))

    # normalize is special due to --no-normalize
    if "normalize" in common:
        if common["normalize"] is False:
            argv.append("--no-normalize")
        elif common["normalize"] is True:
            argv.append("--normalize")

    for t in (common.get("ortholog_type") or []):
        argv.extend(["--ortholog-type", str(t)])

    # targets -> flags
    if subcmd == "genes":
        genes = targets.get("gene_ids") or []
        if genes:
            add_opt("--gene-ids", ",".join(map(str, genes)))
        add_bool("--one-family", targets.get("one_family", False))
        if targets.get("gene_id_file") is not None:
            add_opt("--gene-id-file", targets["gene_id_file"])

    elif subcmd == "fam-map":
        if targets.get("fam_map") is None:
            raise SystemExit("targets.fam_map required for subcommand=fam-map")
        add_opt("--fam-map", targets["fam_map"])
        if targets.get("only_fams"):
            add_opt("--only-fams", ",".join(map(str, targets["only_fams"])))

    elif subcmd == "random":
        if targets.get("universe_file") is None:
            raise SystemExit("targets.universe_file required for subcommand=random")
        add_opt("--universe-file", targets["universe_file"])
        if targets.get("exclude_file") is not None:
            add_opt("--exclude-file", targets["exclude_file"])
        for k, flag in [("n_fams", "--n-fams"), ("fam_size", "--fam-size"), ("seed", "--seed")]:
            if targets.get(k) is not None:
                add_opt(flag, targets[k])
        add_bool("--allow-overlap", targets.get("allow_overlap", False))

    return argv

# ----------------------------
# main
# ----------------------------

def main(argv=None):
    # --- pre-parse only --config ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    ns, rest = pre.parse_known_args(argv)

    parser = build_parser()  # your existing level-1 parser (with run genes/fam-map/random)

    if ns.config is not None:
        cfg = load_cfg(ns.config)
        argv2 = cfg_to_argv(cfg)   # <-- ignore rest completely
        args = parser.parse_args(argv2)
    else:
        args = parser.parse_args(argv)

    builder = TARGET_BUILDERS[args.targets_cmd]
    targets = builder(args)
    run_pipeline(args, targets)



if __name__ == "__main__":
    main()


"""
----------------------------
Example config.yml (genes)
----------------------------

run:
  subcommand: genes

common:
  background_emb: Paper/data/embed/human_idrs_esm1b.csv
  ortholog_emb: Paper/data/embed/orthologs/
  out_results: ./FND_out.csv
  out_matrix: ./matrices/cdk_matrix.npz
  overwrite: false
  metric: cosine
  normalize: true
  dtype: float32
  chunk_size: 2000
  n_jobs: 1
  id_col: 0
  embedding_cols: null
  filter_orthologs: true
  homolog_annotation: Paper/data/annotations/homologs.tsv
  ortholog_type: [ortholog_one2one]
  species_threshold: 0.5
  proteome_species_count_threshold: 1
  min_orthologs: 10
  verbose: true

targets:
  gene_ids: [CDK1, CDK2, CDK3]
  one_family: true

Run:
  python nn_divergence_cli_level1_with_config.py --config config.yml
Override:
  python nn_divergence_cli_level1_with_config.py --config config.yml --out-results out2.csv --min-orthologs 20

"""
