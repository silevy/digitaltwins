#!/usr/bin/env python

"""
summary_sim.py

Reads various Y_u... and Y_c files from the 'simulated_data' subfolder,
computes summary statistics, writes results into a 'summary_stats.txt'
file inside the 'simulated_data/summary_stats' subfolder, and prints
the same summaries to the terminal.
"""

import os
import numpy as np
import datetime

now = datetime.datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

def calculate_summary_stats(arr):
    """
    Given a NumPy array, compute a dictionary of summary stats:
    shape, min, max, number of unique values, mean, std.
    """
    stats = {}
    stats["shape"] = arr.shape
    stats["min"] = float(arr.min())
    stats["max"] = float(arr.max())
    # For large arrays, be mindful that np.unique can be expensive.
    stats["unique_count"] = int(np.unique(arr).size)
    stats["mean"] = float(arr.mean())
    stats["std"] = float(arr.std())
    unique_vals = np.unique(arr)
    stats["unique_count"] = len(unique_vals)
    
    return stats

def main():
    # Folder where .npy files reside
    data_folder = "simulated_data"
    # Folder to store summary statistics
    summary_folder = os.path.join(data_folder, "summary_stats")
    # Create summary folder if not already there
    os.makedirs(summary_folder, exist_ok=True)

    # We will write a single summary file
    summary_file_path = os.path.join(summary_folder, f"\n summary_stats_{date_time}.txt")

    # List of files to summarize:
    # (You could also do a pattern match, e.g., all that start with 'Y_u_' or 'Y_c')
    files_to_summarize = [
        "Y_u_1_10.npy",
        "Y_u_1_11.npy",
        "Y_u_1_5.npy",
        "Y_u_2_10.npy",
        "Y_u_2_11.npy",
        "Y_u_2_5.npy",
        "Y_u_3_10.npy",
        "Y_u_3_11.npy",
        "Y_u_3_5.npy",
        # "c_5_base.npy",
        "c_5.npy",
        # "c_10_base.npy",
        "c_10.npy",
        # "c_11_base.npy",
        "c_11.npy",
        "u_1.npy",
        "u_2.npy",
        "u_3.npy"
        # "Y_c_1_optim.npy",
        # "Y_c_2_optim.npy",
        # "Y_c_3_optim.npy",
        # "Y_c_1_static.npy",
        # "Y_c_2_static.npy",
        # "Y_c_3_static.npy",

    ]

    with open(summary_file_path, "w") as summary_file:

        for filename in files_to_summarize:
            file_path = os.path.join(data_folder, filename)

            if os.path.isfile(file_path):
                arr = np.load(file_path)
                stats = calculate_summary_stats(arr)

                # Start building a text block for this file
                stats_str = (
                    f"File: {filename}\n"
                    f"  Shape          : {stats['shape']}\n"
                    f"  Min            : {stats['min']}\n"
                    f"  Max            : {stats['max']}\n"
                    f"  Unique count   : {stats['unique_count']}\n"
                    f"  Mean           : {stats['mean']:.4f}\n"
                    f"  Std            : {stats['std']:.4f}\n"
                )

                # If unique_count < 30, calculate frequency & percentage
                if stats["unique_count"] < 30:
                    unique_vals, counts = np.unique(arr, return_counts=True)
                    total_count = arr.size

                    stats_str += "  Value Frequencies (unique < 30):\n"
                    for val, cnt in zip(unique_vals, counts):
                        freq_pct = (cnt / total_count) * 100
                        stats_str += f"    Value={val}, Count={cnt}, Percentage={freq_pct:.2f}%\n"

                stats_str += "------------------------------------------------------\n"

                # Print to terminal
                print(stats_str, end="")

                # Write to summary_stats.txt
                summary_file.write(stats_str)

            else:
                # If file is not found
                msg = f"File not found: {filename}\n"
                print(msg)
                summary_file.write(msg)


    
    print(f"\nSummary statistics written to: {summary_file_path}")

if __name__ == "__main__":
    main()