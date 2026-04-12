import os
import zipfile
import pandas as pd
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def extract_datasets():
    """Unzip all attached datasets from data/raw into their respective subfolders."""
    print("Initiating extraction pipeline for local ZIP files...")
    
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        print("No zip files found to extract.")
        return

    for zip_path in zip_files:
        # Create a specific directory for each zip based on its stem
        extract_dir = RAW_DIR / zip_path.stem
        
        if extract_dir.exists():
            print(f"Skipping {zip_path.name} - already extracted.")
            continue
            
        print(f"Extracting {zip_path.name} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Successfully extracted {zip_path.name}")
        except zipfile.BadZipFile:
            print(f"ERROR: {zip_path.name} is corrupted or not a valid zip file.")

def build_unified_dataframe():
    """
    Placeholder logic to iterate through the extracted Empatica E4 output formats 
    and merge them into standardized parquet formats inside data/processed.
    """
    print("Beginning dataset normalization and EDA/BVP alignment...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # The extraction logic will utilize the EmpaticaDataLoader in empatica_loader.py
    # ...

if __name__ == "__main__":
    extract_datasets()
    build_unified_dataframe()
    print("\nPreprocessing script instantiated.")
