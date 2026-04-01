"""Download BEAT2 dataset from HuggingFace."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_beat2(output_dir: str, dry_run: bool = False):
    """Download the BEAT2 dataset.

    Args:
        output_dir: Directory to save the dataset.
        dry_run: If True, download one file and print its structure.
    """
    output_dir = os.path.expandvars(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("=== Dry run: downloading dataset info only ===")
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("H-Liu1997/BEAT2")
        print(f"Dataset: {info.id}")
        print(f"Tags: {info.tags}")
        print(f"Last modified: {info.last_modified}")

        # Download just one file to inspect structure
        print("\nDownloading a sample file to inspect structure...")
        import numpy as np
        sample_path = snapshot_download(
            repo_id="H-Liu1997/BEAT2",
            repo_type="dataset",
            local_dir=os.path.join(output_dir, "_sample"),
            allow_patterns=["**/1_wayne_0_1_1.npz"],
        )
        # Find and inspect the downloaded npz
        for p in Path(sample_path).rglob("*.npz"):
            print(f"\nInspecting: {p.name}")
            data = np.load(str(p), allow_pickle=True)
            print(f"Keys: {list(data.keys())}")
            for key in data.keys():
                arr = data[key]
                if hasattr(arr, "shape"):
                    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    print(f"  {key}: {type(arr)} = {arr}")
            break
        return

    print(f"Downloading BEAT2 dataset to {output_dir}...")
    print("This may take a while (dataset is ~60GB).")

    snapshot_download(
        repo_id="H-Liu1997/BEAT2",
        repo_type="dataset",
        local_dir=output_dir,
    )

    print(f"Download complete. Data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download BEAT2 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expandvars("$WORK/data/beat2_raw"),
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Download one sample and inspect its structure",
    )
    args = parser.parse_args()
    download_beat2(args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
