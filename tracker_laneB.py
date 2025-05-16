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

confidence_threshold = 0.5  # Default threshold for vehicles
bike_confidence_threshold = 0.01  # Higher threshold for bicycle (class 1)
truck_confidence_threshold = 0.84

# Initialize SORT tracker
tracker = Sort(max_age=40, min_hits=1, iou_threshold=0.1)

# Counting line and variables
limitsUp = [70-15, 390-25, 610-15, 389-25]
limitsDown = [60-15, 401+10, 610-15, 399+10]

# Define y-limits for robust line crossing
limitsUp_y_min = limitsUp[1] - 40
limitsUp_y_max = limitsUp[1] + 40
limitsDown_y_min = limitsDown[1] - 40
limitsDown_y_max = limitsDown[1] + 40

# Per-class counting sets
carCountUp = set()
truckCountUp = set()
busCountUp = set()
motorbikeCountUp = set()

carCountDown = set()
truckCountDown = set()
busCountDown = set()
motorbikeCountDown = set()

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

        # Up counting
        if (limitsUp[0] < cx < limitsUp[2]) and (limitsUp_y_min < cy < limitsUp_y_max):
            if cls == 2 and track_id not in carCountUp:
                carCountUp.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountUp:
                motorbikeCountUp.add(track_id)
            elif cls == 5 and track_id not in busCountUp:
                busCountUp.add(track_id)
            elif cls == 7 and track_id not in truckCountUp:
                truckCountUp.add(track_id)
            lineUp_color = (0, 255, 0)  # Turn line green if hit

        # Down counting
        if (limitsDown[0] < cx < limitsDown[2]) and (limitsDown_y_min < cy < limitsDown_y_max):
            if cls == 2 and track_id not in carCountDown:
                carCountDown.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountDown:
                motorbikeCountDown.add(track_id)
            elif cls == 5 and track_id not in busCountDown:
                busCountDown.add(track_id)
            elif cls == 7 and track_id not in truckCountDown:
                truckCountDown.add(track_id)
            lineDown_color = (0, 255, 0)  # Turn line green if hit

        label = f"{classname[cls]} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the counting lines and counts with their current color
    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), lineUp_color, 2)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), lineDown_color, 2)

    # Display all counts at the top of the frame
    cv2.putText(frame, f"UP: Car: {len(carCountUp)}  Truck: {len(truckCountUp)}  Bus: {len(busCountUp)}  Motorbike: {len(motorbikeCountUp)}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    cv2.putText(frame, f"DOWN: Car: {len(carCountDown)}  Truck: {len(truckCountDown)}  Bus: {len(busCountDown)}  Motorbike: {len(motorbikeCountDown)}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    frame_counter += 1

    if frame_counter >= interval_frames:
        # Prepare data
        data = {
            "CarUp": len(carCountUp),
            "TruckUp": len(truckCountUp),
            "BusUp": len(busCountUp),
            "MotorbikeUp": len(motorbikeCountUp),
            "CarDown": len(carCountDown),
            "TruckDown": len(truckCountDown),
            "BusDown": len(busCountDown),
            "MotorbikeDown": len(motorbikeCountDown),
        }
        df = pd.DataFrame([data])
        csv_name = f"vehicles_{interval_counter*interval_seconds}_{(interval_counter+1)*interval_seconds}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {csv_name}")

        # Reset for next interval
        carCountUp.clear()
        truckCountUp.clear()
        busCountUp.clear()
        motorbikeCountUp.clear()
        carCountDown.clear()
        truckCountDown.clear()
        busCountDown.clear()
        motorbikeCountDown.clear()
        frame_counter = 0
        interval_counter += 1

    cv2.imshow("YOLO + SORT", frame)  # Display the processed frame in a window named "YOLO"
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
        break  # If 'q' is pressed, exit the loop

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows


