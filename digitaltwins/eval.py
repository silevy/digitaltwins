import numpy as np
import argparse
import os
import inspect
from scipy import stats

HOME_PATH = os.path.dirname(inspect.getfile(lambda: None))
EVAL_DIR = os.path.join('eval')
RESULTS_DIR = os.path.join('results')
os.makedirs(EVAL_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Load numpy files and print their shapes and first n rows.')
parser.add_argument('--n', default=2, type=int,  help='Number of rows to print')
args = parser.parse_args()

def main():
    n = args.n
    directory = RESULTS_DIR
    files_pairs = [
        ('original_Y_u_1_5.npy', 'post_samples_Y_u_1_5.npy'),
        ('original_Y_u_1_10.npy', 'post_samples_Y_u_1_10.npy'),
        ('original_Y_u_1_11.npy', 'post_samples_Y_u_1_11.npy'),
        ('original_Y_u_2_5.npy', 'post_samples_Y_u_2_5.npy'),
        ('original_Y_u_2_10.npy', 'post_samples_Y_u_2_10.npy'),
        ('original_Y_u_2_11.npy', 'post_samples_Y_u_2_11.npy'),
        ('original_Y_u_3_5.npy', 'post_samples_Y_u_3_5.npy'),
        ('original_Y_u_3_10.npy', 'post_samples_Y_u_3_10.npy'),
        ('original_Y_u_3_11.npy', 'post_samples_Y_u_3_11.npy')
    ]
    
    results_dir = os.path.join(directory, 'mode')
    os.makedirs(results_dir, exist_ok=True)
    
    for original_file, post_samples_file in files_pairs:
        original_path = os.path.join(directory, original_file)
        post_samples_path = os.path.join(directory, post_samples_file)
        
        if os.path.exists(original_path) and os.path.exists(post_samples_path):
            original_data = np.load(original_path)
            post_samples_data = np.load(post_samples_path)
            
            # Compute the mode along the first dimension
            mode_data, _ = stats.mode(post_samples_data, axis=0)
            mode_data = np.squeeze(mode_data)
            
            # Save the mode data
            mode_filename = post_samples_file.replace('post_samples', 'mode')
            mode_path = os.path.join(results_dir, mode_filename)
            np.save(mode_path, mode_data)
            
            print(f"Shape of {original_file}: {original_data.shape}")
            print(f"Shape of {post_samples_file}: {post_samples_data.shape}")
            print(f"Shape of {mode_filename}: {mode_data.shape}")
            print(f"First {n} rows of {original_file}:\n{original_data[:n]}\n")
            print(f"First {n} rows of {mode_filename}:\n{mode_data[:n]}\n")
        else:
            if not os.path.exists(original_path):
                print(f"File {original_file} not found in directory {directory}.")
            if not os.path.exists(post_samples_path):
                print(f"File {post_samples_file} not found in directory {directory}.")


if __name__ == "__main__":
    main()
