import requests
from tqdm import tqdm
import zipfile
import os

dropbox_url = "https://www.dropbox.com/scl/fo/nu00kaibm1dy74lt34ecx/AGX4pLbs5RO1P9dHQgjz13I?rlkey=8wz0y0wfij16q1onwipw19qhx&st=hxpz1b6j&dl=1"
output_file = "nnUNet_results.zip"

print("🚀 Downloading weights...")

response = requests.get(dropbox_url, stream=True)

if response.status_code == 200:
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024

    with open(output_file, "wb") as file, tqdm(
        desc="📥 Progress",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size):
            file.write(chunk)
            bar.update(len(chunk))

    print("✅ Download complete!")
    
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall("nnUNet_results")
    print("🎉 Extraction complete!")
else:
    print(f"😵 Failed to download. Status code: {response.status_code}")