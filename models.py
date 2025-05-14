import cv2
from ultralytics import YOLO
import math
import numpy as np
from sort import Sort
import pandas as pd
import requests
import os
import sys
import time

def download_file(url, filename):
    """Download a file with progress tracking"""
    print(f"Downloading {filename} from {url}...")
    start_time = time.time()
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)")
                        sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed:.1f} seconds!")
        return True
    
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return False


def main():
    models = {
        # Vehicle-Specific Models
        "1": {
            "name": "YOLOv8 BDD100K Vehicles (10 classes)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-bdd100k.pt",
            "filename": "yolov8x_bdd100k.pt",
            "classes": ["car", "bus", "truck", "motorcycle", "bicycle", "rider", "train"]
        },
        "2": {
            "name": "YOLOv5 UA-DETRAC Traffic Vehicles",
            "url": "https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/releases/download/v0.1/best.pt",
            "filename": "yolov5_ua-detrac.pt",
            "classes": ["car", "bus", "van", "others"]
        },
        "3": {
            "name": "YOLOv8 KITTI 3D Vehicles",
            "url": "https://github.com/ruhyadi/vehicle-detection-yolov8/releases/download/v0.0/vehicle_kitti_v0_last.pt",
            "filename": "yolov8_kitti3d.pt",
            "classes": ["car", "truck", "van", "tram"]
        },

        # General Purpose Models with Vehicle Classes
        "4": {
            "name": "YOLOv9 COCO Vehicles",
            "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt",
            "filename": "yolov9_coco.pt",
            "classes": ["car", "truck", "bus", "motorcycle"]
        },
        "5": {
            "name": "YOLOv6 NVIDIA Vehicles",
            "url": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt",
            "filename": "yolov6_nvidia.pt",
            "classes": ["car", "truck", "van", "emergency"]
        },
        "6": {
            "name": "YOLOv7 Traffic Surveillance",
            "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
            "filename": "yolov7_traffic.pt",
            "classes": ["vehicle", "pedestrian", "cyclist"]
        }
    }

    print("Available Vehicle Detection Models:")
    for key, model in models.items():
        print(f"[{key}] {model['name']}")
        print(f"   Classes: {', '.join(model['classes'])}")
        print(f"   Filename: {model['filename']}\n")

    choice = input("Enter model number to download (1-6): ")

    if choice in models:
        model = models[choice]
        if download_file(model["url"], model["filename"]):
            print(f"\nModel Info:")
            print(f"Name: {model['name']}")
            print(f"Classes: {', '.join(model['classes'])}")
            print(f"Usage Example:")
            print(f"model = YOLO('{model['filename']}')")
            print(f"results = model.predict(source, classes={[i for i in range(len(model['classes']))]})")
    else:
        print("Invalid selection!")


main()