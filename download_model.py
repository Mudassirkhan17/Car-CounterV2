import requests
import os
import sys
import time

def download_file(url, filename):
    """
    Download a file from a URL to a specified filename
    with progress tracking
    """
    print(f"Downloading {filename} from {url}...")
    start_time = time.time()
    
    # Stream the download to handle large files properly
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Open file for binary write
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)")
                        sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed:.1f} seconds!")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}")
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def main():
    models = {
        "1": {
            "name": "YOLOv5 BDD100K",
            "url": "https://github.com/williamhyin/yolov5s_bdd100k/releases/download/v1.0/yolov5s_bdd.pt",
            "filename": "yolov5s_bdd.pt"
        },
        "2": {
            "name": "YOLOv8 KITTI",
            "url": "https://github.com/ruhyadi/vehicle-detection-yolov8/releases/download/v0.0/vehicle_kitti_v0_last.pt",
            "filename": "vehicle_kitti_v0_last.pt"
        }
    }
    
    print("Available models to download:")
    for key, model in models.items():
        print(f"{key}. {model['name']} -> {model['filename']}")
    
    choice = input("\nEnter the number of the model to download (default: 1): ") or "1"
    
    if choice in models:
        model = models[choice]
        filename = model["filename"]
        url = model["url"]
        
        # Try primary URL first
        if download_file(url, filename):
            print(f"\nSuccessfully downloaded {filename}!")
            print(f"File saved to: {os.path.abspath(filename)}")
        else:
            print(f"\nFailed to download {filename}. Please try again later or check your internet connection.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 