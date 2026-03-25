#!/usr/bin/env python3
import argparse
from pathlib import Path

from Paper.src.distances.current.neighbour_divergence import run_gene_divergence_pipeline, run_fam_divergence_pipeline, run_random_divergence_pipeline


# ---------- File-friendly ArgumentParser ----------
class FileArgsParser(argparse.ArgumentParser):
    """
    Supports @config.txt files with:
      - comments starting with '#'
      - blank lines
      - '--flag value' or '--flag=value'
      - 'flag=value' (auto-adds leading '--')
      - lone flags like '--random'
    """
    def convert_arg_line_to_args(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"):
            return []
        # allow "--flag value" or multiple tokens on a line
        if " " in line and "=" not in line:
            # e.g., `--genes CDK12,CDK13`
            return line.split()
        # allow "--flag=value" or "flag=value"
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k.startswith("-"):
                k = f"--{k}"
            return [k, v]
        # allow lone flags like "--random"
        if not line.startswith("-"):
            line = f"--{line}"
        return [line]


# ---------- Common parents ----------
def make_common_divergence_parent():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--segment", choices=["idr", "domain"], default="idr")
    p.add_argument("--dist-type", choices=["between", "within"], default="between")
    p.add_argument("--input-file", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--outpath", type=Path, required=True)
    p.add_argument("--emb-path", type=Path)
    return p

def make_index_window_parent():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--start", "-s", type=int, default=0)
    p.add_argument("--stop", "-e", type=int, default=588)
    return p


# ---------- Builders ----------
def add_fam_divergence_cmd(sub, *parents):
    p = sub.add_parser("fam-divergence", parents=list(parents),
                       help="Run divergence across protein families (or subset).")
    p.add_argument("--fam-ids", help='Comma-separated family IDs, e.g. "PF00069,PF00536".')
    p.set_defaults(func=cmd_fam_divergence)

def add_random_divergence_cmd(sub, *parents):
    p = sub.add_parser("random-divergence", parents=list(parents),
                       help="Run divergence on a random sample of segments.")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--n-sample-size", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--random", action="store_true",
                   help="(Optional) Explicit switch also works when using CLI directly.")
    p.set_defaults(func=cmd_random_divergence)

def add_genes_divergence_cmd(sub, *parents):
    p = sub.add_parser("genes-divergence", parents=list(parents),
                       help="Run divergence for a specific set of genes.")
    p.add_argument("--genes", required=True,
                   help='Comma-separated genes, e.g. "CDK12,CDK13".')
    p.set_defaults(func=cmd_genes_divergence)


# ---------- Top-level parser ----------
def build_parser() -> argparse.ArgumentParser:
    parser = FileArgsParser(
        prog="NN_divergence_analysis",
        description="CLI for family, random, and genes-based divergence runs.",
        fromfile_prefix_chars='@',   # <-- enables @config.txt
    )
    # Optional global: allow a config file AFTER the subcommand too
    # Users can do: prog fam-divergence @family.txt

    sub = parser.add_subparsers(dest="cmd", required=True)

    common = make_common_divergence_parent()
    idxwin = make_index_window_parent()

    add_fam_divergence_cmd(sub, common, idxwin)
    add_random_divergence_cmd(sub, common)
    add_genes_divergence_cmd(sub, common)

    return parser


# ---------- Helpers ----------
def _split_csv_list(s: str | None):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------- Handlers (wire to your functions) ----------
def cmd_fam_divergence(args: argparse.Namespace) -> int:
    if args.start < 0 or args.stop < 0 or args.stop < args.start:
        raise SystemExit("Invalid range: ensure 0 <= --start <= --stop.")
    if args.segment == "domain" and args.input_file is None:
        raise SystemExit("--segment domain requires --input-file.")
    if not args.emb_path:
        raise SystemExit("--emb-path is required for family divergence.")
    
    fam_ids = _split_csv_list(args.fam_ids)

    print("run_fam_divergence", vars(args))
    return run_fam_divergence(
        human_embed_path=args.emb_path,
        out_dir=args.outpath,
        fam_ids=fam_ids,
        dist_type=args.dist_type,
        ortho_embed_dir=args.input_dir,
        start_idx=args.start,
        stop_idx=args.stop
        #segment=args.segment,
    )

def cmd_random_divergence(args: argparse.Namespace) -> int:
    # Allow config file to omit an explicit "--random" line; the command implies it
    if args.segment == "domain" and args.input_file is None:
        raise SystemExit("--segment domain requires --input-file for random mode.")
    
    print("run_calc_random_divergence", vars(args))

    return run_calc_random_divergence(
        out_path=args.outpath,
        sample_size=args.sample_size,
        n_samples = args.n_sample_size,
        chunk_size=args.chunk_size,
        full_dict=True,
        select_spp=None,
        dist_type=args.dist_type,
        segment=args.segment,
        #input_file=args.input_file,
        ortho_dir=args.input_dir,
        embed_path=args.emb_path,
    )

def cmd_genes_divergence(args: argparse.Namespace) -> int:
    genes = _split_csv_list(args.genes)
    if not genes:
        raise SystemExit("No genes parsed from --genes.")
    if not args.emb_path:
        raise SystemExit("--emb-path is required for genes divergence.")
    if args.segment == "domain" and args.input_file is None:
        raise SystemExit("--segment domain requires --input-file.")
    
    print("run_calc_divergence", vars(args))
    return run_calc_divergence(
        gene_ids=genes,
        out_path=args.outpath,
        dist_type=args.dist_type,
        segment=args.segment,
        embed_path=args.emb_path,
        input_file=args.input_file,
        input_dir=args.input_dir,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
