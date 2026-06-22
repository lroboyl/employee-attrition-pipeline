"""
Restore the dataset from the local DVC cache.

This script reads the dataset's .dvc pointer file, locates the corresponding
file in the committed DVC cache, and copies it into the data directory.

It provides a lightweight alternative to running dvc pull, making dataset
restoration more reliable in CI environments where DVC version differences
or remote storage access may be unavailable.
"""

import os
import shutil
import yaml


def restore_dataset():
    """Copy the CSV from .dvc/cache_remote/ into data/ using the pointer file."""

    # Read the .dvc pointer file to get the MD5 hash
    dvc_file = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv.dvc"
    if not os.path.exists(dvc_file):
        raise FileNotFoundError(f"DVC pointer file not found: {dvc_file}")

    with open(dvc_file, "r") as f:
        pointer = yaml.safe_load(f)

    md5 = pointer["outs"][0]["md5"]
    prefix = md5[:2]
    suffix = md5[2:]

    # DVC stores files as .dvc/cache_remote/files/md5/<prefix>/<suffix>
    cache_path = os.path.join(".dvc", "cache_remote", "files", "md5", prefix, suffix)
    dest_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if os.path.exists(dest_path):
        print(f"Dataset already exists at {dest_path}")
        return

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Data not found in DVC cache at {cache_path}. "
            "Make sure .dvc/cache_remote/ is committed to the repository."
        )

    shutil.copy2(cache_path, dest_path)
    print(f"Restored dataset to {dest_path}")


if __name__ == "__main__":
    restore_dataset()