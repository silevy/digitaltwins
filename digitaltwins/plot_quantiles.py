import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate posterior predictive histograms from NumPy files.')
parser.add_argument('--directory', default="results", type=str, help='Directory containing the NumPy files')
parser.add_argument('--output-directory', default="plots", type=str, help='Directory to save the output plots')
args = parser.parse_args()

def main():

    directory = args.directory
    output_directory = args.output_directory

    # Pairs of (observed-data file, posterior-simulated-data file)
    files_pairs = [
        ('original_Y_u_1_5.npy',  'post_samples_Y_u_1_5.npy'),
        ('original_Y_u_1_10.npy', 'post_samples_Y_u_1_10.npy'),
        ('original_Y_u_1_11.npy', 'post_samples_Y_u_1_11.npy'),
        ('original_Y_u_2_5.npy',  'post_samples_Y_u_2_5.npy'),
        ('original_Y_u_2_10.npy', 'post_samples_Y_u_2_10.npy'),
        ('original_Y_u_2_11.npy', 'post_samples_Y_u_2_11.npy'),
        ('original_Y_u_3_5.npy',  'post_samples_Y_u_3_5.npy'),
        ('original_Y_u_3_10.npy', 'post_samples_Y_u_3_10.npy'),
        ('original_Y_u_3_11.npy', 'post_samples_Y_u_3_11.npy')
    ]
    
    os.makedirs(output_directory, exist_ok=True)
    
    for original_file, post_samples_file in files_pairs:
        original_path = os.path.join(directory, original_file)
        post_samples_path = os.path.join(directory, post_samples_file)
        
        if os.path.exists(original_path) and os.path.exists(post_samples_path):
            # -------------------------------------------------
            # 1) Load and flatten observed data
            # -------------------------------------------------
            original_data = np.load(original_path).flatten()

            # -------------------------------------------------
            # 2) Load posterior-simulated data
            #    post_samples_data assumed shape:
            #    (n_draws, number_of_simulated_points)
            #
            #    If needed, we can flatten each draw below.
            # -------------------------------------------------
            post_samples_data = np.load(post_samples_path)

            # Determine the range of scale points (assuming integer scale from 0..max)
            num_scale_points = int(original_data.max() + 1)

            # -------------------------------------------------
            # Observed bin counts
            # -------------------------------------------------
            original_counts = np.bincount(original_data.astype(int),
                                          minlength=num_scale_points)

            # -------------------------------------------------
            # 3) For each draw in the posterior data, compute bin counts
            #    shape: (n_draws, num_scale_points)
            # -------------------------------------------------
            n_draws = post_samples_data.shape[0]
            bin_counts_per_draw = np.zeros((n_draws, num_scale_points), dtype=int)

            for d in range(n_draws):
                # Flatten if needed (e.g., if each draw is N x T shape):
                single_draw = post_samples_data[d].flatten()
                # Bin it:
                draw_counts = np.bincount(single_draw.astype(int),
                                          minlength=num_scale_points)
                bin_counts_per_draw[d, :] = draw_counts

            # -------------------------------------------------
            # 4) Compute quantiles of the bin counts across draws
            #    e.g. 2.5, 25, 50, 75, 97.5
            #    Result shape: (5, num_scale_points)
            # -------------------------------------------------
            q_levels = [2.5, 25, 50, 75, 97.5]
            bin_count_quantiles = np.percentile(bin_counts_per_draw, q_levels, axis=0)

            # -------------------------------------------------
            # 5) Plot
            # -------------------------------------------------
            fig, ax = plt.subplots(figsize=(7, 4))

            # Observed data in black step
            ax.step(range(num_scale_points), original_counts,
                    where='mid', color='black', label='Observed')

            # Fill between 2.5% and 97.5%
            ax.fill_between(range(num_scale_points),
                            bin_count_quantiles[0, :],  # 2.5%
                            bin_count_quantiles[-1, :], # 97.5%
                            step='mid',
                            color='red', alpha=0.2,
                            label='2.5%-97.5%')

            # Fill between 25% and 75%
            ax.fill_between(range(num_scale_points),
                            bin_count_quantiles[1, :],  # 25%
                            bin_count_quantiles[3, :],  # 75%
                            step='mid',
                            color='red', alpha=0.4,
                            label='25%-75%')

            # Median (50%)
            ax.step(range(num_scale_points),
                    bin_count_quantiles[2, :],
                    where='mid',
                    color='red', linestyle='--',
                    label='Median (50%)')

            ax.set_title('Posterior Predictive Check')
            ax.set_xlabel('Scale Points')
            ax.set_ylabel('Counts')
            ax.legend(loc='best')

            # -------------------------------------------------
            # 6) Save the figure
            # -------------------------------------------------
            plot_filename = original_file.replace('original_', 'plot_').replace('.npy', '.png')
            plot_path = os.path.join(output_directory, plot_filename)
            plt.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close()

        else:
            # If files are missing, print a warning
            if not os.path.exists(original_path):
                print(f"File {original_file} not found in directory {directory}.")
            if not os.path.exists(post_samples_path):
                print(f"File {post_samples_file} not found in directory {directory}.")

if __name__ == "__main__":
    main()
