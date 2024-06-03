import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate plots from numpy files.')
parser.add_argument('--directory', default="results", type=str, help='Directory containing the numpy files')
parser.add_argument('--output-directory', default="plots", type=str, help='Directory to save the output plots')

args = parser.parse_args()

def main():

    directory = args.directory
    output_directory = args.output_directory

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
    
    os.makedirs(output_directory, exist_ok=True)
    
    for original_file, post_samples_file in files_pairs:
        original_path = os.path.join(directory, original_file)
        post_samples_path = os.path.join(directory, post_samples_file)
        
        if os.path.exists(original_path) and os.path.exists(post_samples_path):
            original_data = np.load(original_path).flatten()
            post_samples_data = np.load(post_samples_path)
            
            # Determine the number of scale points
            num_scale_points = int(original_data.max() + 1)  # Assuming scale points start from 0
            
            # Create the plot
            fig, ax = plt.subplots()
            
            # Plot the original data (black line)
            original_counts = np.bincount(original_data.astype(int), minlength=num_scale_points)
            ax.step(range(num_scale_points), original_counts, where='mid', color='black')
            
            # Plot the post samples data (red histograms)
            post_samples_counts = np.zeros((5, num_scale_points))
            quantiles = np.percentile(post_samples_data, [2.5, 25, 50, 75, 97.5], axis=0)
            
            for q in range(5):
                for i in range(num_scale_points):
                    count = np.sum(quantiles[q] == i)
                    post_samples_counts[q, i] = count
            
            for q in range(5):
                alpha = 0.1 + 0.2 * q
                ax.fill_between(range(num_scale_points), 0, post_samples_counts[q], step='mid', color='red', alpha=alpha)
            
            ax.set_title('Posterior Retrodictive Check')
            ax.set_xlabel('Scale Points')
            ax.set_ylabel('Frequency')
            
            # Save the plot
            plot_filename = original_file.replace('original_', 'plot_').replace('.npy', '.png')
            plot_path = os.path.join(output_directory, plot_filename)
            plt.savefig(plot_path)
            plt.close()
        else:
            if not os.path.exists(original_path):
                print(f"File {original_file} not found in directory {directory}.")
            if not os.path.exists(post_samples_path):
                print(f"File {post_samples_file} not found in directory {directory}.")

if __name__ == "__main__":
    main()