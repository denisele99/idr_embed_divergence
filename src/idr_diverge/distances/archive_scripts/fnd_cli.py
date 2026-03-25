import argparse

class ArgumentParserWithComments(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        arg_line = arg_line.strip()
        if not arg_line or arg_line.startswith("#"):
            return []
        return arg_line.split()

def build_parser():
    p = ArgumentParserWithComments(
        prog="nn-divergence",
        fromfile_prefix_chars='@'
        
    )
    
    
#!/usr/bin/env python3
"""
nn_divergence_cli_level1.py

Level 1 simplification:
- Keep subcommands: run genes | run fam-map | run random
- Remove 3 nearly-identical handlers
- Use ONE shared pipeline runner + a TARGET_BUILDERS registry

This is still a scaffold: replace `compute_divergence(...)` and `load_and_filter_homologs(...)`
with your real logic/schema.
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
    ids: List[str]  # order matches D rows/cols


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

    # TODO: implement species_threshold, proteome_species_count_threshold, min_orthologs using your schema
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
                raise SystemExit("Not enough remaining genes to sample without overlap. Use --allow-overlap or reduce sizes.")
            picks = rng.choice(remaining, size=fam_size, replace=False).tolist()
            used.update(picks)

        targets.append(FamilyTarget(fam_id=f"RANDOM_{i:04d}", genes=picks, meta={"mode": "random", "seed": seed}))
    return targets


# ----------------------------
# Core divergence computation (stub)
# ----------------------------

def compute_divergence(
    matrix: MatrixWithIds,
    targets: List[FamilyTarget],
    homolog_df: Optional[pd.DataFrame],
    verbose: bool,
) -> pd.DataFrame:
    """
    Replace this with your real divergence calculation.
    This stub just demonstrates plumbing per target.
    """
    id_to_idx = {sid: i for i, sid in enumerate(matrix.ids)}
    rows: List[Dict[str, Any]] = []

    for t in targets:
        present = [g for g in t.genes if g in id_to_idx]
        missing = [g for g in t.genes if g not in id_to_idx]

        mean_within = float("nan")
        if len(present) >= 2:
            idxs = [id_to_idx[g] for g in present]
            subD = matrix.D[np.ix_(idxs, idxs)]
            mean_within = float(np.mean(subD))

        rows.append(
            {
                "fam_id": t.fam_id,
                "n_genes_input": len(t.genes),
                "n_present": len(present),
                "n_missing": len(missing),
                "missing_genes": ";".join(missing) if missing else "",
                "mean_within_target_distance": mean_within,
                "homolog_rows_used": int(len(homolog_df)) if homolog_df is not None else 0,
            }
        )
        if verbose:
            print(f"[analysis] {t.fam_id}: present={len(present)} missing={len(missing)}", file=sys.stderr)

    return pd.DataFrame(rows)


# ----------------------------
# Shared pipeline runner (THIS is the Level 1 simplification)
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


def get_or_build_matrix(cfg: CommonConfig) -> MatrixWithIds: #TODO replace with mine
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
    root = argparse.ArgumentParser(prog="nn-divergence", description="Neighbour divergence CLI (Level 1 simplified).")
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


# ---- target builders (now just “args -> targets”) ----

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd != "run":
        raise SystemExit(f"Unknown command: {args.cmd}")

    builder = TARGET_BUILDERS.get(args.targets_cmd)
    if builder is None:
        raise SystemExit(f"Unknown targets subcommand: {args.targets_cmd}")

    targets = builder(args)
    run_pipeline(args, targets)


if __name__ == "__main__":
    main()
