import cv2  # OpenCV library for computer vision tasks
from ultralytics import YOLO  # YOLO object detection model from Ultralytics
import math  # For mathematical operations
import numpy as np
from sort import Sort
import pandas as pd

model = YOLO("yolov5su.pt")  # Load the pre-trained YOLOv5 nano model

classname = ["Person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]  # List of class names that YOLO can detect

vehicle_classes = set([2, 3, 5, 7])  # car, motorcycle, bus, truck

# cap = cv2.VideoCapture(0)  # Initialize webcam capture (0 refers to the default camera)
video_path = "00002.MTS"
cap = cv2.VideoCapture(video_path) 
# cap.set(3, 640)  # Set webcam width to 640 pixels
# cap.set(4, 640)  # Set webcam height to 640 pixels

confidence_threshold = 0.12  # Default threshold for vehicles
bike_confidence_threshold = 0.001  # Higher threshold for bicycle (class 1)
car_confidence_threshold = 0.35
truck_confidence_threshold = 0.84

# Initialize SORT tracker
tracker = Sort(max_age=60, min_hits=2, iou_threshold=0.1)


#laneA [900, 368+10, 1103, 393+10]
# laneB  [410, 416, 788, 596]
# lane c [159, 357-10, 279, 407-10]
# lane e temporary [960+35, 363, 1025+35, 355], [1044, 361, 1150, 357]
# lane e [975+35, 363+7, 1100+35, 355+7]


# lanea1 [609+20, 404, 1024+20, 403]
# lanea2 [609+90, 404+30, 1024+90, 403+30]
# laneb1 [240-30, 412, 624-30, 411]

# Counting line and variables
laneA = [240-30, 412-3, 624-30, 411-3]  # (x1, y1, x2, y2)
laneB = [240-90, 412+30, 624-90, 411+30]  # (514+100, 396+40, 1058+100, 387+40)


# Define y-limits for robust line crossing
laneA_y_min = laneA[1] - 17
laneA_y_max = laneA[1] + 17
laneB_y_min = laneB[1] - 17
laneB_y_max = laneB[1] + 17

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
lineA_color = (0, 0, 255)    # Red in BGR
lineB_color = (0, 0, 255)  # Red in BGR

interval_seconds = 30  # 2 minutes
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
        elif cls == 2:
            threshold = car_confidence_threshold
        elif cls == 3:
            threshold = bike_confidence_threshold
        elif cls == 0:
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
    lineA_color = (0, 0, 255)
    lineB_color = (0, 0, 255)

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

        # Up counting (laneA)
        if (laneA[0] < cx < laneA[2]) and (laneA_y_min < cy < laneA_y_max):
            if cls == 2 and track_id not in carCountUp:
                carCountUp.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountUp:
                motorbikeCountUp.add(track_id)
            elif cls == 5 and track_id not in busCountUp:
                busCountUp.add(track_id)
            elif cls == 7 and track_id not in truckCountUp:
                truckCountUp.add(track_id)
            lineA_color = (0, 255, 0)  # Turn line green if hit

        # Down counting (laneB)
        if (laneB[0] < cx < laneB[2]) and (laneB_y_min < cy < laneB_y_max):
            if cls == 2 and track_id not in carCountDown:
                carCountDown.add(track_id)
            elif cls == 3 and track_id not in motorbikeCountDown:
                motorbikeCountDown.add(track_id)
            elif cls == 5 and track_id not in busCountDown:
                busCountDown.add(track_id)
            elif cls == 7 and track_id not in truckCountDown:
                truckCountDown.add(track_id)
            lineB_color = (0, 255, 0)  # Turn line green if hit

        label = f"{classname[cls]} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the counting lines and counts with their current color
    cv2.line(frame, (laneA[0], laneA[1]), (laneA[2], laneA[3]), lineA_color, 2)
    cv2.line(frame, (laneB[0], laneB[1]), (laneB[2], laneB[3]), lineB_color, 2)

    # Display all counts at the top of the frame
    cv2.putText(frame, f"UP: Car: {len(carCountUp)}  Truck: {len(truckCountUp)}  Bus: {len(busCountUp)}  Motorbike: {len(motorbikeCountUp)}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1)
    cv2.putText(frame, f"DOWN: Car: {len(carCountDown)}  Truck: {len(truckCountDown)}  Bus: {len(busCountDown)}  Motorbike: {len(motorbikeCountDown)}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1)

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


