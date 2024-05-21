import os
import requests
import shutil
import zipfile

# Function to download individual files
def download_file(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Function to download and extract a folder from a ZIP archive
def download_and_extract_folder(url, extract_to, folder_in_zip, final_dest):
    response = requests.get(url)
    response.raise_for_status()
    zip_path = 'temp.zip'
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

    # Ensure the final destination directory exists
    os.makedirs(final_dest, exist_ok=True)

    # Move the specified folder to the final destination
    full_folder_path = os.path.join(extract_to, folder_in_zip)
    for item in os.listdir(full_folder_path):
        shutil.move(os.path.join(full_folder_path, item), final_dest)

    # Clean up temporary files and directories
    shutil.rmtree(extract_to)

# Specify files and folders to download
files_to_download = [
    {'url': 'https://raw.githubusercontent.com/yang-song/score_sde_pytorch/main/sde_lib.py', 'save_as': 'sde_lib.py'},
    {'url': 'https://raw.githubusercontent.com/yang-song/score_sde_pytorch/main/utils.py', 'save_as': 'utils.py'},
]

folders_to_download = [
    {'url': 'https://github.com/yang-song/score_sde_pytorch/archive/refs/heads/main.zip', 'extract_to': './temp_extract', 'folder_in_zip': 'score_sde_pytorch-main/models', 'final_dest': './models'},
]

# Download files
for file_info in files_to_download:
    download_file(file_info['url'], file_info['save_as'])

# Download folders
for folder_info in folders_to_download:
    zip_url = folder_info['url']
    extract_temp = folder_info['extract_to']
    folder_in_zip = folder_info['folder_in_zip']
    final_dest = folder_info['final_dest']
    
    # Download and extract the repository archive
    download_and_extract_folder(zip_url, extract_temp, folder_in_zip, final_dest)
