
#conda activate denise_trn
#cd /home/moseslab/denise/scripts
#python esm_embed_clean.py /home/moseslab/denise/IDR_LM/data/sequences/allyeast_scer_filtered.fasta allyeast_idrs_scer_esm

import os
import subprocess
import sys
import argparse

# # Read all files in
# data_DIR = '/home/moseslab/denise/embeddings/data_to_embed/'
# result_DIR = '/home/moseslab/denise/embeddings/RES/'

# # Ensure the script is called with the required arguments
# if len(sys.argv) < 2:
#     raise ValueError("Please provide the embedding type as an argument: ['esm', 'IDR_LM', 'IDR_LM_randominit']")

# embed_type = sys.argv[1]  # ['esm', 'IDR_LM', 'IDR_LM_randominit']

# files = os.listdir(data_DIR)

# if embed_type == 'esm':
#     for file in files:
#         out_file = file.split('/')[-1].split('.')[0]
#         print(out_file, '...')
#         out_path = os.path.join(result_DIR, out_file)
#         cmd = f'python /home/moseslab/denise/scripts/esm_embed_clean.py {os.path.join(data_DIR, file)} {out_path}_esm'
#         subprocess.run(cmd, shell=True, capture_output=True)
#         print(out_path)

# elif embed_type == 'IDR_LM': #TODO still need to edit to change out_path naming or else it won't go to the RES directory
#     if len(sys.argv) < 3:
#         raise ValueError("Please provide the model file path as the second argument for 'IDR_LM' embedding type.")
    
#     model_file = sys.argv[2]
    
#     for file in files:
#         out_file = file.split('/')[-1].split('.')[0]
#         print(out_file, '...')
#         out_file = out_file + model_file.split('_')[-1]
#         out_path = os.path.join(result_DIR, out_file)
#         #cmd = f'python /home/moseslab/denise/IDR_LM/src/test/get_embeddings_idr_batch_2.py {os.path.join(data_DIR, file)} {model_file} {out_path}_IDRLM'
#         cmd = f'python /home/moseslab/denise/IDR_LM/src/test/get_embeddings_idr_batch_2.py {os.path.join(data_DIR, file)} {model_file} {out_file}_IDRLM'
#         subprocess.run(cmd, shell=True, capture_output=True)
#         print(out_path)

# elif embed_type == 'IDR_LM_random':
#     for file in files:
#         out_file = file.split('/')[-1].split('.')[0]
#         print(out_file, '...')
#         out_path = os.path.join(result_DIR, out_file)
#         cmd = f'python /home/moseslab/denise/IDR_LM/src/test/get_embeddings_idr_batch_2.py {os.path.join(data_DIR, file)}'
#         subprocess.run(cmd, shell=True, capture_output=True)
#         print(out_path)

# else:
#     raise ValueError("Choose one of ['esm', 'IDR_LM', 'IDR_LM_randominit']")


#Command = python run_embed_10-24.py esm



import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run embedding scripts for protein sequences.")
    
    parser.add_argument(
        "--embed-type", 
        required=True, 
        choices=["esm", "IDR_LM", "IDR_LM_random"],
        help="Type of embedding to generate: ['esm', 'IDR_LM', 'IDR_LM_random']"
    )
    
    parser.add_argument(
        "--model-file", 
        help="Path to the model file (required if --embed-type is 'IDR_LM')"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="/home/moseslab/denise/embeddings/data_to_embed/",
        help="Directory containing input FASTA files"
    )
    
    parser.add_argument(
        "--result-dir", 
        default="/home/moseslab/denise/embeddings/RES/",
        help="Directory to write embedding results"
    )

    args = parser.parse_args()

    data_DIR = args.data_dir
    result_DIR = args.result_dir
    embed_type = args.embed_type
    files = os.listdir(data_DIR)

    if embed_type == "esm":
        for file in files:
            out_file = os.path.splitext(file)[0]
            print(out_file, '...')
            out_path = os.path.join(result_DIR, out_file)
            cmd = f'python /home/moseslab/denise/scripts/esm_embed_clean.py {os.path.join(data_DIR, file)} {out_path}_esm'
            subprocess.run(cmd, shell=True, capture_output=True)
            print(out_path)

    elif embed_type == "IDR_LM":
        if not args.model_file:
            raise ValueError("Please provide the --model-file argument when using 'IDR_LM' embedding type.")

        model_file = args.model_file

        for file in files:
            out_file = os.path.splitext(file)[0]
            print(out_file, '...')
            suffix = model_file.split('_')[-1]
            out_file = f"{out_file}_{suffix}"
            out_path = os.path.join(result_DIR, out_file)
            cmd = f'python /home/moseslab/denise/IDR_LM/src/test/get_embeddings_idr_batch_2.py {os.path.join(data_DIR, file)} {model_file} {out_file}_IDRLM'
            subprocess.run(cmd, shell=True, capture_output=True)
            print(out_path)

    elif embed_type == "IDR_LM_random":
        for file in files:
            out_file = os.path.splitext(file)[0]
            print(out_file, '...')
            out_path = os.path.join(result_DIR, out_file)
            cmd = f'python /home/moseslab/denise/IDR_LM/src/test/get_embeddings_idr_batch_2.py {os.path.join(data_DIR, file)}'
            subprocess.run(cmd, shell=True, capture_output=True)
            print(out_path)

if __name__ == "__main__":
    main()


#Examples
#python run_embedding.py --embed-type esm --data-dir X --result-dir X
#python run_embedding.py --embed-type IDR_LM --model-file /path/to/model_checkpoint.pt