import urllib.request
import zipfile
import io
import os
import shutil

url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
print(f"Downloading Eigen from {url}...")

# Create 3rdparty dir
if not os.path.exists("3rdparty"):
    os.makedirs("3rdparty")

try:
    with urllib.request.urlopen(url) as response:
        data = response.read()
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall("3rdparty")
            
            # Find the extracted folder name (usually eigen-3.4.0)
            extracted_name = [n for n in z.namelist() if '/' in n][0].split('/')[0]
            src = os.path.join("3rdparty", extracted_name)
            dst = os.path.join("3rdparty", "eigen")
            
            if os.path.exists(dst):
                print(f"Removing existing {dst}...")
                shutil.rmtree(dst)
                
            print(f"Renaming {src} to {dst}...")
            os.rename(src, dst)
            
    print("Eigen setup complete.")

except Exception as e:
    print(f"Failed to download/extract Eigen: {e}")
