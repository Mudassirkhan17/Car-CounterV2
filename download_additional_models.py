import requests
import os
import sys
import time
import gdown  # For Google Drive downloads

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

def download_gdrive(file_id, output):
    """Download a file from Google Drive"""
    try:
        print(f"Downloading {output} from Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        print(f"Successfully downloaded {output}!")
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        return False

def main():
    # Install gdown if not already installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system("pip install gdown")
        import gdown
    
    models = {
        "1": {
            "name": "YOLOv8s Traffic (Ultralytics)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-traffic.pt",
            "filename": "yolov8s-traffic.pt",
            "type": "direct"
        },
        "2": {
            "name": "YOLOv8n BDD100K (Ultralytics)",
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-bdd100k.pt",
            "filename": "yolov8n-bdd100k.pt",
            "type": "direct"
        },
        "3": {
            "name": "YOLOv5 DeepSORT Vehicle Tracking",
            "url": "1-8Xm3eUMMJF5XNiF649kqnqoYeWhv3kT",
            "filename": "yolov5_deepsort_vehicle.pt",
            "type": "gdrive"
        },
        "4": {
            "name": "YOLOv5 Medium Traffic (5 classes)",
            "url": "1zw0rR7iSfobJ9CwPXe2-YqvjrSmjzt_T",
            "filename": "yolov5m_traffic_5class.pt",
            "type": "gdrive"
        },
        "5": {
            "name": "YOLOv8 Nano Traffic (5 classes)",
            "url": "1OSU5g3yz-IliqMFQhfbmVp0UA6e5nKcT",
            "filename": "yolov8n_traffic_5class.pt",
            "type": "gdrive"
        }
    }
    
    print("Additional Vehicle Detection Models:")
    for key, model in models.items():
        print(f"{key}. {model['name']} -> {model['filename']}")
    
    while True:
        choice = input("\nEnter the number of the model to download (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
            
        if choice in models:
            model = models[choice]
            filename = model["filename"]
            url = model["url"]
            download_type = model["type"]
            
            if download_type == "direct":
                if download_file(url, filename):
                    print(f"\nSuccessfully downloaded {filename}!")
                    print(f"File saved to: {os.path.abspath(filename)}")
                else:
                    print(f"\nFailed to download {filename}. Please try again later.")
            elif download_type == "gdrive":
                if download_gdrive(url, filename):
                    print(f"File saved to: {os.path.abspath(filename)}")
                else:
                    print(f"\nFailed to download {filename} from Google Drive.")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 