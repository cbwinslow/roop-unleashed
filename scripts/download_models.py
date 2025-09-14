#!/usr/bin/env python3
import os
import urllib.request
import hashlib

models = {
    "inswapper_128.onnx": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "hash": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
    }
}

def download_model(name, info):
    model_path = f"/app/models/{name}"
    if os.path.exists(model_path):
        print(f"Model {name} already exists")
        return
    
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(info["url"], model_path)
    
    # Verify hash if provided
    if "hash" in info:
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != info["hash"]:
            os.remove(model_path)
            raise ValueError(f"Hash mismatch for {name}")
    
    print(f"Downloaded {name} successfully")

if __name__ == "__main__":
    os.makedirs("/app/models", exist_ok=True)
    for name, info in models.items():
        try:
            download_model(name, info)
        except Exception as e:
            print(f"Failed to download {name}: {e}")