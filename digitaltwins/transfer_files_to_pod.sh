#!/bin/bash

# Define the local directory containing the files
LOCAL_DIR="~/Documents/workspace"

# Define the remote server details
REMOTE_USER="root"
REMOTE_HOST="213.173.110.31"
REMOTE_DIR="/workspace"

# List of files to transfer
FILES=("model.py" "util.py" "main.py" "sim.py" "inout.py")

# Loop through each file and transfer it using scp
for FILE in "${FILES[@]}"; do
  scp "${LOCAL_DIR}/${FILE}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
done

echo "Files transferred successfully to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
