#!/bin/bash

# Define the remote server details
REMOTE_USER="root"
REMOTE_HOST="213.173.110.31"
REMOTE_DIR="/workspace"

# Define the local destination directory
LOCAL_DIR="/Users/samlevy/Documents/Documents - Samuel’s MacBook Pro/DigitalTwin/Runpod"

# List of files to copy
FILES=("model.py" "util.py" "main.py" "sim.py" "inout.py", "post.py")

# Loop through each file and copy it using scp
for FILE in "${FILES[@]}"; do
  scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${FILE}" "${LOCAL_DIR}/"
done


# Directories to transfer back from the pod to the local machine
DIRECTORIES=("results" "fit_results" "simulated_data")

# Loop through each directory and copy it back using scp
echo "Copying directories back from the pod..."
for DIR in "${DIRECTORIES[@]}"; do
  scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${DIR}" "${LOCAL_DIR}/"
done

echo "Files copied successfully to ${LOCAL_DIR}"
