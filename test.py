import cv2  # OpenCV library for computer vision tasks
from ultralytics import YOLO  # YOLO object detection model from Ultralytics
import math  # For mathematical operations
import numpy as np
from sort import Sort
import pandas as pd

model = YOLO("yolov5lu.pt")  # Load the pre-trained YOLOv5 nano model

classname = ["Person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]  # List of class names that YOLO can detect

vehicle_classes = set([2, 3, 5, 7])  # car, motorcycle, bus, truck

start_seconds = 40  # Change this to where you want to start (in seconds)

# cap = cv2.VideoCapture(0)  # Initialize webcam capture (0 refers to the default camera)
video_path = "location2.MTS"
cap = cv2.VideoCapture(video_path) 
# cap.set(3, 640)  # Set webcam width to 640 pixels
# cap.set(4, 640)  # Set webcam height to 640 pixels

cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)

confidence_threshold = 0.05  # Default threshold for vehicles
bike_confidence_threshold = 0.000  # Higher threshold for bicycle (class 1)
car_confidence_threshold = 0.40
truck_confidence_threshold = 0.84

# Initialize SORT tracker
tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.1)


#laneA [900, 368+10, 1103, 393+10]
# laneB  [410, 416, 788, 596]
# lane c [159, 357-10, 279, 407-10]
# lane e temporary [960+35, 363, 1025+35, 355], [1044, 361, 1150, 357]
# lane e [975+35, 363+7, 1100+35, 355+7]


# lanea1 [609+20, 404, 1024+20, 403]
# lanea2 [609+90, 404+30, 1024+90, 403+30]
# laneb1 [240-30, 412, 624-30, 411]

# Counting line and variables
laneA = [1000, 328, 1158, 351]  # (x1, y1, x2, y2)
laneB = [599-35, 396+7, 1071-35, 395+7]  # (514+100, 396+40, 1058+100, 387+40)
laneC = [199, 404, 315, 444]  # Placeholder coordinates for laneC
laneD = [136-60, 407, 311-90, 462]  # Placeholder coordinates for laneD
laneE = [167, 359+10, 252, 338+10]  # Placeholder coordinates for laneE
laneF = [167+60, 359+8, 252+60, 338+8]  # Placeholder coordinates for laneF
laneG = [224, 324, 408-60, 322]  # Placeholder coordinates for laneG

# laneD = [0, 0, 0, 0]
# laneC = [0, 0, 0, 0]
# laneB = [0, 0, 0, 0]

# Define y-limits for robust line crossing
laneA_y_min = laneA[1] - 12
laneA_y_max = laneA[1] + 12
laneB_y_min = laneB[1] - 23
laneB_y_max = laneB[1] + 23
laneC_y_min = laneC[1] - 17
laneC_y_max = laneC[1] + 17
laneD_y_min = laneD[1] - 17
laneD_y_max = laneD[1] + 17
laneE_y_min = laneE[1] - 17
laneE_y_max = laneE[1] + 17
laneF_y_min = laneF[1] - 22
laneF_y_max = laneF[1] + 22
laneG_y_min = laneG[1] - 17
laneG_y_max = laneG[1] + 17

# Per-class counting sets
laneACarCount = set()
laneATruckCount = set()
laneABusCount = set()
laneAMotorbikeCount = set()

laneBCarCount = set()
laneBTruckCount = set()
laneBBusCount = set()
laneBMotorbikeCount = set()

laneCCarCount = set()
laneCTruckCount = set()
laneCBusCount = set()
laneCMotorbikeCount = set()

laneDCarCount = set()
laneDTruckCount = set()
laneDBusCount = set()
laneDMotorbikeCount = set()

laneECarCount = set()
laneETruckCount = set()
laneEBusCount = set()
laneEMotorbikeCount = set()

laneFCarCount = set()
laneFTruckCount = set()
laneFBusCount = set()
laneFMotorbikeCount = set()

laneGCarCount = set()
laneGTruckCount = set()
laneGBusCount = set()
laneGMotorbikeCount = set()

# Add these before the while loop
lineA_color = (0, 0, 255)    # Red in BGR
lineB_color = (0, 0, 255)  # Red in BGR
lineC_color = (0, 0, 255)  # Red in BGR
lineD_color = (0, 0, 255)  # Red in BGR
lineE_color = (0, 0, 255)  # Red in BGR
lineF_color = (0, 0, 255)  # Red in BGR
lineG_color = (0, 0, 255)  # Red in BGR

interval_seconds = 300  # 2 minutes
fps = cap.get(cv2.CAP_PROP_FPS)
interval_frames = int(interval_seconds * fps)
frame_counter = 0
interval_counter = start_seconds // interval_seconds

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
    lineC_color = (0, 0, 255)
    lineD_color = (0, 0, 255)
    lineE_color = (0, 0, 255)
    lineF_color = (0, 0, 255)
    lineG_color = (0, 0, 255)

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

        # LaneA counting
        if (laneA[0] < cx < laneA[2]) and (laneA_y_min < cy < laneA_y_max):
            if cls == 2 and track_id not in laneACarCount:
                laneACarCount.add(track_id)
            elif cls == 3 and track_id not in laneAMotorbikeCount:
                laneAMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneABusCount:
                laneABusCount.add(track_id)
            elif cls == 7 and track_id not in laneATruckCount:
                laneATruckCount.add(track_id)
            lineA_color = (0, 255, 0)  # Turn line green if hit

        # LaneB counting
        if (laneB[0] < cx < laneB[2]) and (laneB_y_min < cy < laneB_y_max):
            if cls == 2 and track_id not in laneBCarCount:
                laneBCarCount.add(track_id)
            elif cls == 3 and track_id not in laneBMotorbikeCount:
                laneBMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneBBusCount:
                laneBBusCount.add(track_id)
            elif cls == 7 and track_id not in laneBTruckCount:
                laneBTruckCount.add(track_id)
            lineB_color = (0, 255, 0)  # Turn line green if hit

        # LaneC counting
        if (laneC[0] < cx < laneC[2]) and (laneC_y_min < cy < laneC_y_max):
            if cls == 2 and track_id not in laneCCarCount:
                laneCCarCount.add(track_id)
            elif cls == 3 and track_id not in laneCMotorbikeCount:
                laneCMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneCBusCount:
                laneCBusCount.add(track_id)
            elif cls == 7 and track_id not in laneCTruckCount:
                laneCTruckCount.add(track_id)
            lineC_color = (0, 255, 0)  # Turn line green if hit

        # LaneD counting
        if (laneD[0] < cx < laneD[2]) and (laneD_y_min < cy < laneD_y_max):
            if cls == 2 and track_id not in laneDCarCount:
                laneDCarCount.add(track_id)
            elif cls == 3 and track_id not in laneDMotorbikeCount:
                laneDMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneDBusCount:
                laneDBusCount.add(track_id)
            elif cls == 7 and track_id not in laneDTruckCount:
                laneDTruckCount.add(track_id)
            lineD_color = (0, 255, 0)  # Turn line green if hit

        # LaneE counting
        if (laneE[0] < cx < laneE[2]) and (laneE_y_min < cy < laneE_y_max):
            if cls == 2 and track_id not in laneECarCount:
                laneECarCount.add(track_id)
            elif cls == 3 and track_id not in laneEMotorbikeCount:
                laneEMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneEBusCount:
                laneEBusCount.add(track_id)
            elif cls == 7 and track_id not in laneETruckCount:
                laneETruckCount.add(track_id)
            lineE_color = (0, 255, 0)  # Turn line green if hit

        # LaneF counting
        if (laneF[0] < cx < laneF[2]) and (laneF_y_min < cy < laneF_y_max):
            if cls == 2 and track_id not in laneFCarCount:
                laneFCarCount.add(track_id)
            elif cls == 3 and track_id not in laneFMotorbikeCount:
                laneFMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneFBusCount:
                laneFBusCount.add(track_id)
            elif cls == 7 and track_id not in laneFTruckCount:
                laneFTruckCount.add(track_id)
            lineF_color = (0, 255, 0)  # Turn line green if hit

        # LaneG counting
        if (laneG[0] < cx < laneG[2]) and (laneG_y_min < cy < laneG_y_max):
            if cls == 2 and track_id not in laneGCarCount:
                laneGCarCount.add(track_id)
            elif cls == 3 and track_id not in laneGMotorbikeCount:
                laneGMotorbikeCount.add(track_id)
            elif cls == 5 and track_id not in laneGBusCount:
                laneGBusCount.add(track_id)
            elif cls == 7 and track_id not in laneGTruckCount:
                laneGTruckCount.add(track_id)
            lineG_color = (0, 255, 0)  # Turn line green if hit

        label = f"{classname[cls]} ID:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the counting lines and counts with their current color
    cv2.line(frame, (laneA[0], laneA[1]), (laneA[2], laneA[3]), lineA_color, 2)
    cv2.line(frame, (laneB[0], laneB[1]), (laneB[2], laneB[3]), lineB_color, 2)
    cv2.line(frame, (laneC[0], laneC[1]), (laneC[2], laneC[3]), lineC_color, 2)
    cv2.line(frame, (laneD[0], laneD[1]), (laneD[2], laneD[3]), lineD_color, 2)
    cv2.line(frame, (laneE[0], laneE[1]), (laneE[2], laneE[3]), lineE_color, 2)
    cv2.line(frame, (laneF[0], laneF[1]), (laneF[2], laneF[3]), lineF_color, 2)
    cv2.line(frame, (laneG[0], laneG[1]), (laneG[2], laneG[3]), lineG_color, 2)

    # Display all counts at the top of the frame
    cv2.putText(frame, f"LaneA: Car: {len(laneACarCount)}  Truck: {len(laneATruckCount)}  Bus: {len(laneABusCount)}  Motorbike: {len(laneAMotorbikeCount)}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 1)
    cv2.putText(frame, f"LaneB: Car: {len(laneBCarCount)}  Truck: {len(laneBTruckCount)}  Bus: {len(laneBBusCount)}  Motorbike: {len(laneBMotorbikeCount)}",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1)
    cv2.putText(frame, f"LaneC: Car: {len(laneCCarCount)}  Truck: {len(laneCTruckCount)}  Bus: {len(laneCBusCount)}  Motorbike: {len(laneCMotorbikeCount)}",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1)
    cv2.putText(frame, f"LaneD: Car: {len(laneDCarCount)}  Truck: {len(laneDTruckCount)}  Bus: {len(laneDBusCount)}  Motorbike: {len(laneDMotorbikeCount)}",
                (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1)
    cv2.putText(frame, f"LaneE: Car: {len(laneECarCount)}  Truck: {len(laneETruckCount)}  Bus: {len(laneEBusCount)}  Motorbike: {len(laneEMotorbikeCount)}",
                (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 1)
    cv2.putText(frame, f"LaneF: Car: {len(laneFCarCount)}  Truck: {len(laneFTruckCount)}  Bus: {len(laneFBusCount)}  Motorbike: {len(laneFMotorbikeCount)}",
                (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 1)
    cv2.putText(frame, f"LaneG: Car: {len(laneGCarCount)}  Truck: {len(laneGTruckCount)}  Bus: {len(laneGBusCount)}  Motorbike: {len(laneGMotorbikeCount)}",
                (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 255), 1)

    frame_counter += 1

    if frame_counter >= interval_frames:
        # Prepare data
        data = {
            "LaneA_Car": len(laneACarCount),
            "LaneA_Truck": len(laneATruckCount),
            "LaneA_Bus": len(laneABusCount),
            "LaneA_Motorbike": len(laneAMotorbikeCount),
            "LaneB_Car": len(laneBCarCount),
            "LaneB_Truck": len(laneBTruckCount),
            "LaneB_Bus": len(laneBBusCount),
            "LaneB_Motorbike": len(laneBMotorbikeCount),
            "LaneC_Car": len(laneCCarCount),
            "LaneC_Truck": len(laneCTruckCount),
            "LaneC_Bus": len(laneCBusCount),
            "LaneC_Motorbike": len(laneCMotorbikeCount),
            "LaneD_Car": len(laneDCarCount),
            "LaneD_Truck": len(laneDTruckCount),
            "LaneD_Bus": len(laneDBusCount),
            "LaneD_Motorbike": len(laneDMotorbikeCount),
            "LaneE_Car": len(laneECarCount),
            "LaneE_Truck": len(laneETruckCount),
            "LaneE_Bus": len(laneEBusCount),
            "LaneE_Motorbike": len(laneEMotorbikeCount),
            "LaneF_Car": len(laneFCarCount),
            "LaneF_Truck": len(laneFTruckCount),
            "LaneF_Bus": len(laneFBusCount),
            "LaneF_Motorbike": len(laneFMotorbikeCount),
            "LaneG_Car": len(laneGCarCount),
            "LaneG_Truck": len(laneGTruckCount),
            "LaneG_Bus": len(laneGBusCount),
            "LaneG_Motorbike": len(laneGMotorbikeCount),
        }
        df = pd.DataFrame([data])
        csv_name = f"vehicles_{interval_counter*interval_seconds}_{(interval_counter+1)*interval_seconds}_location3_card1_00003.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {csv_name}")

        # Reset for next interval
        laneACarCount.clear()
        laneATruckCount.clear()
        laneABusCount.clear()
        laneAMotorbikeCount.clear()
        laneBCarCount.clear()
        laneBTruckCount.clear()
        laneBBusCount.clear()
        laneBMotorbikeCount.clear()
        laneCCarCount.clear()
        laneCTruckCount.clear()
        laneCBusCount.clear()
        laneCMotorbikeCount.clear()
        laneDCarCount.clear()
        laneDTruckCount.clear()
        laneDBusCount.clear()
        laneDMotorbikeCount.clear()
        laneECarCount.clear()
        laneETruckCount.clear()
        laneEBusCount.clear()
        laneEMotorbikeCount.clear()
        laneFCarCount.clear()
        laneFTruckCount.clear()
        laneFBusCount.clear()
        laneFMotorbikeCount.clear()
        laneGCarCount.clear()
        laneGTruckCount.clear()
        laneGBusCount.clear()
        laneGMotorbikeCount.clear()
        frame_counter = 0
        interval_counter += 1

    cv2.imshow("YOLO + SORT", frame)  # Display the processed frame in a window named "YOLO"
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
        break  # If 'q' is pressed, exit the loop

# After the while loop, save any remaining counts for the last partial interval
if frame_counter > 0:
    data = {
        "LaneA_Car": len(laneACarCount),
        "LaneA_Truck": len(laneATruckCount),
        "LaneA_Bus": len(laneABusCount),
        "LaneA_Motorbike": len(laneAMotorbikeCount),
        "LaneB_Car": len(laneBCarCount),
        "LaneB_Truck": len(laneBTruckCount),
        "LaneB_Bus": len(laneBBusCount),
        "LaneB_Motorbike": len(laneBMotorbikeCount),
        "LaneC_Car": len(laneCCarCount),
        "LaneC_Truck": len(laneCTruckCount),
        "LaneC_Bus": len(laneCBusCount),
        "LaneC_Motorbike": len(laneCMotorbikeCount),
        "LaneD_Car": len(laneDCarCount),
        "LaneD_Truck": len(laneDTruckCount),
        "LaneD_Bus": len(laneDBusCount),
        "LaneD_Motorbike": len(laneDMotorbikeCount),
        "LaneE_Car": len(laneECarCount),
        "LaneE_Truck": len(laneETruckCount),
        "LaneE_Bus": len(laneEBusCount),
        "LaneE_Motorbike": len(laneEMotorbikeCount),
        "LaneF_Car": len(laneFCarCount),
        "LaneF_Truck": len(laneFTruckCount),
        "LaneF_Bus": len(laneFBusCount),
        "LaneF_Motorbike": len(laneFMotorbikeCount),
        "LaneG_Car": len(laneGCarCount),
        "LaneG_Truck": len(laneGTruckCount),
        "LaneG_Bus": len(laneGBusCount),
        "LaneG_Motorbike": len(laneGMotorbikeCount),
    }
    df = pd.DataFrame([data])
    csv_name = f"vehicles_{interval_counter*interval_seconds}_{interval_counter*interval_seconds + frame_counter/fps:.0f}_location3_card1_00003.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved {csv_name} (final partial interval)")

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows


