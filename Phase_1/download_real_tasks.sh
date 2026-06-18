#!/bin/bash
# download_real_tasks.sh
# Downloads the official pre-generated EnactToM tasks from Hugging Face

# Replace this with the exact official dataset repository name.
# Because the official release repository is not publicly indexed, you must supply the exact name 
# provided by the authors (e.g., "ai-habitat/enacttom_tasks" or "UCSB-AI/EnactTom-Benchmark").
HF_REPO="<REPLACE_WITH_DATASET_REPO>"

if [ "$HF_REPO" = "<REPLACE_WITH_DATASET_REPO>" ]; then
    echo "Error: Please edit this script and set HF_REPO to the correct Hugging Face repository name."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please export your HuggingFace token: export HF_TOKEN=..."
    exit 1
fi

TARGET_DIR="Others/EnactTom/data/enacttom/tasks"
mkdir -p "$TARGET_DIR"

echo "Downloading official EnactToM tasks from $HF_REPO to $TARGET_DIR..."

# Use huggingface-cli to download the contents of the dataset directly into the tasks folder.
# This assumes the dataset repository contains the .json files at its root, or within a tarball.
# If they are in a tarball, we will download the tarball and extract it.

huggingface-cli download "$HF_REPO" \
    --repo-type dataset \
    --local-dir "$TARGET_DIR" \
    --local-dir-use-symlinks False \
    --token "$HF_TOKEN"

echo "Download complete! Checking contents of $TARGET_DIR..."
ls -lh "$TARGET_DIR" | head -n 10

# If the downloaded file is a zip or tar.gz, extract it
for archive in "$TARGET_DIR"/*.tar.gz; do
    if [ -f "$archive" ]; then
        echo "Extracting $archive..."
        tar -xzf "$archive" -C "$TARGET_DIR"
        rm "$archive"
    fi
done

for archive in "$TARGET_DIR"/*.zip; do
    if [ -f "$archive" ]; then
        echo "Extracting $archive..."
        unzip -q "$archive" -d "$TARGET_DIR"
        rm "$archive"
    fi
done

echo "Tasks are ready in $TARGET_DIR."
