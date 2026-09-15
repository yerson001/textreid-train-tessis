"""
C. September 2024
Doc: Download PeterPanTheGenius/CUHK-PEDES from HuggingFace
      and convert to the format expected by the training script.

Usage:
    .venv/bin/python scripts/download_hf_dataset.py
"""

import os
import sys
import subprocess


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    output_dir = os.path.join(_project_root, 'data', 'CUHK-PEDES-HF')
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'reid_raw.json')
    if os.path.exists(json_path):
        print(f"Dataset already converted: {json_path}")
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"  Entries: {len(data)}")
        return

    # Step 1: Download HF dataset (run in subprocess to avoid datasets name collision)
    parquet_path = os.path.join(output_dir, "raw.parquet")
    if not os.path.exists(parquet_path):
        print("=" * 60)
        print("Downloading PeterPanTheGenius/CUHK-PEDES from HuggingFace...")
        print(f"Output dir: {output_dir}")
        print("=" * 60)

        dl_script = f"""
import sys, os
_project = os.path.abspath("{_project_root}")
sys.path = [p for p in sys.path if os.path.abspath(p) != _project]
from datasets import load_dataset
ds = load_dataset("PeterPanTheGenius/CUHK-PEDES", split="train")
print(f"Loaded {{len(ds)}} rows.")
ds.to_parquet("{parquet_path}")
print("Parquet saved.")
"""
        venv_python = os.path.join(_project_root, '.venv', 'bin', 'python')
        result = subprocess.run([venv_python, '-c', dl_script], capture_output=True, text=True)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            sys.exit(1)
        print(result.stdout)
    else:
        print(f"Parquet already exists: {parquet_path}")

    # Step 2: Convert to project format (run in clean subprocess)
    print("Converting to project format...")
    convert_script = f"""
import sys, os
sys.path.insert(0, "{_project_root}")
from datasets.cuhkpedes_hf import build_cuhkpedes_hf_anno
build_cuhkpedes_hf_anno("{parquet_path}", "{output_dir}")
print("Conversion done.")
"""
    venv_python = os.path.join(_project_root, '.venv', 'bin', 'python')
    result = subprocess.run([venv_python, '-c', convert_script], capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)
    print(result.stdout)

    print("\nDataset ready!")
    print(f"  - Images: {os.path.join(output_dir, 'imgs/all/')}")
    print(f"  - Annotations: {json_path}")


if __name__ == '__main__':
    main()
