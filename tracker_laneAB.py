import cv2  # OpenCV library for computer vision tasks
from ultralytics import YOLO  # YOLO object detection model from Ultralytics
import math  # For mathematical operations
import numpy as np
from sort import Sort
import pandas as pd

model = YOLO("yolov5n.pt")  # Load the pre-trained YOLOv5 nano model

classname = ["Person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]  # List of class names that YOLO can detect

vehicle_classes = set([2, 3, 5, 7])  # car, motorcycle, bus, truck

# cap = cv2.VideoCapture(0)  # Initialize webcam capture (0 refers to the default camera)
video_path = "location1.MTS"
cap = cv2.VideoCapture(video_path) 
# cap.set(3, 640)  # Set webcam width to 640 pixels
# cap.set(4, 640)  # Set webcam height to 640 pixels

confidence_threshold = 0.6  # Default threshold for vehicles
bike_confidence_threshold = 0.05  # Higher threshold for bicycle (class 1)
truck_confidence_threshold = 0.84

# Initialize SORT tracker
tracker = Sort(max_age=40, min_hits=1, iou_threshold=0.1)

# Lane A (right side)
limitsUpA = [614, 386, 1238, 377]
limitsDownA = [614, 431, 1258, 422]

# Lane B (left side, example values)
limitsUpB = [100, 386, 400, 377]
limitsDownB = [100, 431, 400, 422]

# Define y-limits for robust line crossing
limitsUpA_y_min = limitsUpA[1] - 30
limitsUpA_y_max = limitsUpA[1] + 30
limitsDownA_y_min = limitsDownA[1] - 30
limitsDownA_y_max = limitsDownA[1] + 30

limitsUpB_y_min = limitsUpB[1] - 30
limitsUpB_y_max = limitsUpB[1] + 30
limitsDownB_y_min = limitsDownB[1] - 30
limitsDownB_y_max = limitsDownB[1] + 30

# Per-class counting sets
carCountUpA = set()
truckCountUpA = set()
busCountUpA = set()
motorbikeCountUpA = set()

carCountDownA = set()
truckCountDownA = set()
busCountDownA = set()
motorbikeCountDownA = set()

# Lane B
carCountUpB = set()
truckCountUpB = set()
busCountUpB = set()
motorbikeCountUpB = set()

carCountDownB = set()
truckCountDownB = set()
busCountDownB = set()
motorbikeCountDownB = set()

# Add these before the while loop
lineUp_color = (0, 0, 255)    # Red in BGR
lineDown_color = (0, 0, 255)  # Red in BGR

interval_seconds = 60  # 2 minutes
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(interval_seconds * fps)
frame_counter = 0
interval_counter = 0

while True:  # Start an infinite loop to continuously process video frames
    ret, frame = cap.read()  # Read a frame from the webcam (ret is True if successful)
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model(frame)  # Run the YOLO model on the current frame

    detections = []
    class_ids = []
    for box in results[0].boxes:  # Loop through the results
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Set threshold: use bike_confidence_threshold for bicycle, truck_confidence_threshold for truck, else default
        if cls == 1:
            threshold = bike_confidence_threshold
        elif cls == 7:
            threshold = truck_confidence_threshold
        else:
            threshold = confidence_threshold

        if cls in vehicle_classes and conf > threshold:
            x1,y1,x2,y2 = box.xyxy[0]  # Get coordinates of the bounding box (x1,y1 is top-left, x2,y2 is bottom-right)
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  # Convert coordinates to integers
            detections.append([x1, y1, x2, y2, conf])
            class_ids.append(cls)

    # Convert detections to numpy array for SORT
    if len(detections) > 0:
        dets = np.array(detections)
    else:
        dets = np.empty((0, 5))

    # Update tracker and get tracks
    tracks = tracker.update(dets)

    # Reset line colors to red at the start of each frame
    lineUp_color = (0, 0, 255)
    lineDown_color = (0, 0, 255)

    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, track)
        # Find the class for this detection (by IoU or nearest box)
        # For simplicity, match by order if detections and tracks are aligned
        # In practice, you may want to use a more robust association
        cls = None
        for det, cid in zip(detections, class_ids):
            if abs(x1 - det[0]) < 10 and abs(y1 - det[1]) < 10 and abs(x2 - det[2]) < 10 and abs(y2 - det[3]) < 10:
                cls = cid
                break
        if cls is None:
            cls = 2  # Default to car if not found

        # Calculate the center of the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # UpA counting
        if (limitsUpA[0] < cx < limitsUpA[2]) and (limitsUpA_y_min < cy < limitsUpA_y_max):
            if cls == 2 and track_id not in carCountUpA:
                carCountUpA.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountUpA:
                motorbikeCountUpA.add(track_id)
            elif cls == 5 and track_id not in busCountUpA:
                busCountUpA.add(track_id)
            elif cls == 7 and track_id not in truckCountUpA:
                truckCountUpA.add(track_id)
            lineUp_color = (0, 255, 0)  # Turn line green if hit

        # UpB counting
        if (limitsUpB[0] < cx < limitsUpB[2]) and (limitsUpB_y_min < cy < limitsUpB_y_max):
            if cls == 2 and track_id not in carCountUpB:
                carCountUpB.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountUpB:
                motorbikeCountUpB.add(track_id)
            elif cls == 5 and track_id not in busCountUpB:
                busCountUpB.add(track_id)
            elif cls == 7 and track_id not in truckCountUpB:
                truckCountUpB.add(track_id)

        # DownA counting
        if (limitsDownA[0] < cx < limitsDownA[2]) and (limitsDownA_y_min < cy < limitsDownA_y_max):
            if cls == 2 and track_id not in carCountDownA:
                carCountDownA.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountDownA:
                motorbikeCountDownA.add(track_id)
            elif cls == 5 and track_id not in busCountDownA:
                busCountDownA.add(track_id)
            elif cls == 7 and track_id not in truckCountDownA:
                truckCountDownA.add(track_id)

        # DownB counting
        if (limitsDownB[0] < cx < limitsDownB[2]) and (limitsDownB_y_min < cy < limitsDownB_y_max):
            if cls == 2 and track_id not in carCountDownB:
                carCountDownB.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountDownB:
                motorbikeCountDownB.add(track_id)
            elif cls == 5 and track_id not in busCountDownB:
                busCountDownB.add(track_id)
            elif cls == 7 and track_id not in truckCountDownB:
                truckCountDownB.add(track_id)

        label = f"{classname[cls]} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the counting lines and counts with their current color
    cv2.line(frame, (limitsUpA[0], limitsUpA[1]), (limitsUpA[2], limitsUpA[3]), (255, 0, 255), 2)
    cv2.line(frame, (limitsDownA[0], limitsDownA[1]), (limitsDownA[2], limitsDownA[3]), (0, 255, 255), 2)
    cv2.line(frame, (limitsUpB[0], limitsUpB[1]), (limitsUpB[2], limitsUpB[3]), (255, 0, 255), 2)
    cv2.line(frame, (limitsDownB[0], limitsDownB[1]), (limitsDownB[2], limitsDownB[3]), (0, 255, 255), 2)

    # Display all counts at the top of the frame
    cv2.putText(frame, f"UP A: Car: {len(carCountUpA)}  Truck: {len(truckCountUpA)}  Bus: {len(busCountUpA)}  Motorbike: {len(motorbikeCountUpA)}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"UP B: Car: {len(carCountUpB)}  Truck: {len(truckCountUpB)}  Bus: {len(busCountUpB)}  Motorbike: {len(motorbikeCountUpB)}",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"DOWN A: Car: {len(carCountDownA)}  Truck: {len(truckCountDownA)}  Bus: {len(busCountDownA)}  Motorbike: {len(motorbikeCountDownA)}",
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"DOWN B: Car: {len(carCountDownB)}  Truck: {len(truckCountDownB)}  Bus: {len(busCountDownB)}  Motorbike: {len(motorbikeCountDownB)}",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    frame_counter += 1

    if frame_counter >= interval_frames:
        # Prepare data
        data = {
            "CarUpA": len(carCountUpA),
            "TruckUpA": len(truckCountUpA),
            "BusUpA": len(busCountUpA),
            "MotorbikeUpA": len(motorbikeCountUpA),
            "CarDownA": len(carCountDownA),
            "TruckDownA": len(truckCountDownA),
            "BusDownA": len(busCountDownA),
            "MotorbikeDownA": len(motorbikeCountDownA),
            "CarUpB": len(carCountUpB),
            "TruckUpB": len(truckCountUpB),
            "BusUpB": len(busCountUpB),
            "MotorbikeUpB": len(motorbikeCountUpB),
            "CarDownB": len(carCountDownB),
            "TruckDownB": len(truckCountDownB),
            "BusDownB": len(busCountDownB),
            "MotorbikeDownB": len(motorbikeCountDownB),
        }
        df = pd.DataFrame([data])
        csv_name = f"vehicles_{interval_counter*interval_seconds}_{(interval_counter+1)*interval_seconds}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {csv_name}")

        # Reset for next interval
        carCountUpA.clear()
        truckCountUpA.clear()
        busCountUpA.clear()
        motorbikeCountUpA.clear()
        carCountDownA.clear()
        truckCountDownA.clear()
        busCountDownA.clear()
        motorbikeCountDownA.clear()
        carCountUpB.clear()
        truckCountUpB.clear()
        busCountUpB.clear()
        motorbikeCountUpB.clear()
        carCountDownB.clear()
        truckCountDownB.clear()
        busCountDownB.clear()
        motorbikeCountDownB.clear()
        frame_counter = 0
        interval_counter += 1

    cv2.imshow("YOLO + SORT", frame)  # Display the processed frame in a window named "YOLO"
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
        break  # If 'q' is pressed, exit the loop

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows


