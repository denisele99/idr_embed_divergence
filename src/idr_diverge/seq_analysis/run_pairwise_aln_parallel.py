import os
import pandas as pd
import subprocess
from itertools import combinations
import argparse
import ast

import sys.path.append('/home/moseslab/denise/Thesis/src/')
from utils.helper_functions import read_fasta


def run_pairwise_aln_parallel(input_fasta_path, list_ids, out_path, n_processes=4, dry_run=False):
    """
    Run pairwise alignments in parallel for a list of ID sets.

    Parameters:
        input_fasta_path (str): Path to the input FASTA file.
        list_ids (list of lists): List of lists of IDs to align pairwise.
        out_path (str): Path to save alignment outputs.
        n_processes (int): Number of parallel processes to use.
        dry_run (bool): If True, print commands but do not execute.
    """
    processes = []
    all_pair_combinations = []

    for ids in list_ids:
        pair_combinations = [f'{a}-{b}' for a, b in combinations(ids, 2)]
        all_pair_combinations.extend(pair_combinations)

    # Split into chunks for parallel processing
    chunk_size = max(1, len(all_pair_combinations) // n_processes)
    pair_list = [all_pair_combinations[i:i + chunk_size] for i in range(0, len(all_pair_combinations), chunk_size)]

    for pairs in pair_list:
        cmd = ['python', '/home/moseslab/denise/Thesis/src/seq_analysis/run_pairwise_aln.py', input_fasta_path, out_path] + pairs
        print(f"Running: {' '.join(cmd)}")

        if not dry_run:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        stdout, stderr = process.communicate()
        print(f"Output of {process.args}: {stdout.decode().strip()}")
        if stderr:
            print(f"Error in {process.args}: {stderr.decode().strip()}")


def main():
    parser = argparse.ArgumentParser(description="Run batch pairwise alignments in parallel.")
    parser.add_argument("--input_fasta", required=True, help="Path to fasta containing sequences and corresponding Uniprot IDs.")
    parser.add_argument("--outpath", required=True, help="Output path to write alignment results.")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--dry_run", action='store_true', help="Print commands without executing them.")

    args = parser.parse_args()

    fasta_dict = read_fasta(args.input_fasta)
    
    
    selected_ids = fasta_dict.keys()

    # Split into 20 chunks
    chunk_size = max(1, len(selected_ids) // args.n_processes)
    selected_ids_chunk = [selected_ids[i:i + chunk_size] for i in range(0, len(selected_ids), chunk_size)]

    for chunk in selected_ids_chunk:
        # if 'IDR' in name:
        #     input_fasta = '/home/moseslab/denise/IDR_LM/data/sequences/IDRs/human_idrs/human_idrs_pos_updated_labels.fasta'
        # elif 'PFAM' in name:
        #     input_fasta = '/home/moseslab/denise/IDR_LM/data/sequences/NON_IDRs/human_non_IDRs/human_PFAM_updated_labels.fasta'
        # else:
        #     print(f"Skipping {name} (unknown type)")
        #     continue

        run_pairwise_aln_parallel(
            input_fasta_path=args.input_fasta,
            list_ids=[chunk],
            out_path=args.out_path,
            n_processes=args.n_processes,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()


# import subprocess
# import time

# def run_scripts_in_parallel(script_path, fasta_path, out_path, pairs, max_parallel=4):
#     processes = []
#     for i, pair in enumerate(pairs):
#         cmd = ['python', script_path, fasta_path, out_path, pair]
#         print(f"Launching: {' '.join(cmd)}")
#         processes.append(subprocess.Popen(cmd))

#         # If we've hit max_parallel, wait for all to finish
#         if len(processes) == max_parallel:
#             for p in processes:
#                 p.wait()
#             processes = []

#     # Wait for any remaining processes
#     for p in processes:
#         p.wait()

# # Example usage
# if __name__ == "__main__":
#     pair_list = ['gene1,gene2', 'gene3,gene4', 'gene5,gene6']  # Or load from file
#     run_scripts_in_parallel(
#         script_path='run_pairwise_aln.py',
#         fasta_path='sequences.fasta',
#         out_path='output.csv',
#         pairs=pair_list,
#         max_parallel=4
#     )
