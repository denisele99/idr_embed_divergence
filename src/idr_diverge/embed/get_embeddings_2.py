from pathlib import Path
import argparse
import subprocess


DEFAULT_DATA_DIR = Path("/home/moseslab/denise/embeddings/data_to_embed/")
DEFAULT_RESULT_DIR = Path("/home/moseslab/denise/embeddings/RES/")

ESM_SCRIPT = Path("/home/moseslab/denise/Paper/src/idr_diverge/embed/esm_embed.py")
IDR_LM_SCRIPT = Path("/home/moseslab/denise/Paper/src/idr_diverge/embed/idrlm_embed.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run embedding scripts for protein sequence files."
    )

    parser.add_argument(
        "--embed-type",
        required=True,
        choices=["esm", "IDR_LM", "IDR_LM_random"],
        help="Embedding type to generate.",
    )

    parser.add_argument(
        "--model-file",
        type=Path,
        help="Path to the trained model checkpoint. Required for --embed-type IDR_LM.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing input FASTA files.",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory where embedding outputs will be written.",
    )

    return parser.parse_args()


def get_input_files(data_dir: Path) -> list[Path]:
    """
    Return sequence files from the input directory.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {data_dir}")

    files = sorted(
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".fa", ".fasta", ".faa", ".txt"}
    )

    if not files:
        raise FileNotFoundError(f"No FASTA-like files found in: {data_dir}")

    return files


def ensure_output_dir(result_dir: Path) -> None:
    """
    Create the output directory if it does not already exist.
    """
    result_dir.mkdir(parents=True, exist_ok=True)


def extract_model_tag(model_file: Path) -> str:
    """
    Create a short tag from the model filename for output naming.
    Example:
        checkpoint_epoch_10.pt -> pt
        model_lowlr_epoch10 -> epoch10
    """
    stem = model_file.stem
    parts = stem.split("_")
    return parts[-1] if parts else stem


def run_command(cmd: list[str]) -> None:
    """
    Run a subprocess command and print useful debugging output if it fails.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Command failed:")
        print(" ".join(cmd))
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    if result.stdout:
        print(result.stdout.strip())


def run_esm_embeddings(input_files: list[Path], result_dir: Path) -> None:
    """
    Run the ESM embedding script on each input file.
    """
    for input_file in input_files:
        output_prefix = result_dir / f"{input_file.stem}_esm"
        print(f"Processing {input_file.name} -> {output_prefix}")

        cmd = [
            "python",
            str(ESM_SCRIPT),
            str(input_file),
            str(output_prefix),
        ]
        run_command(cmd)


def run_idr_lm_embeddings(
    input_files: list[Path],
    result_dir: Path,
    model_file: Path,
    random_init: bool = False,
) -> None:
    """
    Run the IDR_LM embedding script on each input file.

    If random_init=True, call the script in its random/uninitialized mode.
    """
    if not random_init and model_file is None:
        raise ValueError(
            "--model-file is required when --embed-type is 'IDR_LM'."
        )

    if model_file is not None and not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    model_tag = extract_model_tag(model_file) if model_file else "random"

    for input_file in input_files:
        output_name = f"{input_file.stem}_{model_tag}_IDRLM"
        output_prefix = result_dir / output_name
        print(f"Processing {input_file.name} -> {output_prefix}")

        if random_init:
            cmd = [
                "python",
                str(IDR_LM_SCRIPT),
                str(input_file),
            ]
        else:
            cmd = [
                "python",
                str(IDR_LM_SCRIPT),
                str(input_file),
                str(model_file),
                str(output_name),
            ]

        run_command(cmd)


def main():
    args = parse_args()

    ensure_output_dir(args.result_dir)
    input_files = get_input_files(args.data_dir)

    if args.embed_type == "esm":
        run_esm_embeddings(input_files, args.result_dir)

    elif args.embed_type == "IDR_LM":
        run_idr_lm_embeddings(
            input_files=input_files,
            result_dir=args.result_dir,
            model_file=args.model_file,
            random_init=False,
        )

    elif args.embed_type == "IDR_LM_random":
        run_idr_lm_embeddings(
            input_files=input_files,
            result_dir=args.result_dir,
            model_file=None,
            random_init=True,
        )


if __name__ == "__main__":
    main()