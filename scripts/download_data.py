import os
import requests
import zipfile
import subprocess

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

def download_zenodo_record(record_id, extract=True):
    print(f"Fetching metadata for Zenodo record {record_id}...")
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Failed to fetch Zenodo API for {record_id}")
        return
    
    data = response.json()
    files = data.get('files', [])
    record_dir = os.path.join(RAW_DATA_DIR, f"zenodo_{record_id}")
    os.makedirs(record_dir, exist_ok=True)
    
    for f in files:
        file_url = f.get('links', {}).get('self')
        file_name = f.get('key') or f.get('id')
        if not file_url or not file_name:
            continue
            
        print(f"Downloading {file_name}...")
        file_path = os.path.join(record_dir, file_name)
        
        # Stream download
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f_out:
                for chunk in r.iter_content(chunk_size=8192): 
                    f_out.write(chunk)
                    
        print(f"Saved to {file_path}")
        
        # Optional extraction for zips
        if extract and file_name.endswith(".zip"):
            print(f"Extracting {file_name}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(record_dir)

def download_physionet():
    print("Downloading PhysioNet Wearable Stress Dataset using wget...")
    # Using wget to gracefully mirror the physionet database
    physionet_dir = os.path.join(RAW_DATA_DIR, "physionet_wearable_stress")
    os.makedirs(physionet_dir, exist_ok=True)
    
    cmd = [
        "wget", "-r", "-N", "-c", "-np", "-nd",
        "-A", "csv,txt,hea,dat",
        "https://physionet.org/files/wearable-stress-exercise/1.0.0/",
        "-P", physionet_dir
    ]
    
    try:
        # Run wget but don't block forever if it's too large, just let it run.
        subprocess.run(cmd, check=True)
        print(f"PhysioNet dataset downloaded to {physionet_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading PhysioNet data: {e}")

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 1. Measurement of mental workload everyday (Zenodo 7936993)
    print("==== Starting Zenodo Dataset Download (7936993) ====")
    download_zenodo_record("7936993", extract=False)  # We can extract manually later if needed
    
    # 2. PhysioNet Stress and Exercise
    print("\n==== Starting PhysioNet Dataset Download ====")
    download_physionet()
    
    print("\nData collection process completed.")
