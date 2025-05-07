import cv2  # OpenCV library for computer vision tasks
from ultralytics import YOLO  # YOLO object detection model from Ultralytics
import math  # For mathematical operations


model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv5 nano model

classname = ["Person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]  # List of class names that YOLO can detect

vehicle_classes = set([2, 3, 5, 7])  # car, motorcycle, bus, truck

# cap = cv2.VideoCapture(0)  # Initialize webcam capture (0 refers to the default camera)
video_path = "location1.MTS"
cap = cv2.VideoCapture(video_path) 
# cap.set(3, 640)  # Set webcam width to 640 pixels
# cap.set(4, 640)  # Set webcam height to 640 pixels


confidence_threshold = 0.45  # Default threshold for vehicles
bike_confidence_threshold = 0.05  # Higher threshold for bicycle (class 1)
truck_confidence_threshold = 0.84  # Higher threshold for truck (class 7)


limitsUp = [614, 396, 1038, 387]
counted_ids = set()
vehicle_up_count = 0


while True:  # Start an infinite loop to continuously process video frames
    ret, frame = cap.read()  # Read a frame from the webcam (ret is True if successful)
    if not ret:
        break

    frame = cv2.resize(frame, (900, 500))

    results = model(frame)  # Run the YOLO model on the current frame

    for box in results[0].boxes:  # Loop through the results
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Set threshold: use bike_confidence_threshold for bicycle, else default
        if cls == 1:
            threshold = bike_confidence_threshold
        elif cls == 7:
            threshold = truck_confidence_threshold
        else:
            threshold = confidence_threshold

        if cls in vehicle_classes and conf > threshold:
            x1,y1,x2,y2 = box.xyxy[0]  # Get coordinates of the bounding box (x1,y1 is top-left, x2,y2 is bottom-right)
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  # Convert coordinates to integers
            w,h = x2-x1,y2-y1  # Calculate width and height of the bounding box

            # Calculate the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw the center
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Check if the center crosses the line and hasn't been counted yet
            if (limitsUp[0] < cx < limitsUp[2]) and (abs(cy - limitsUp[1]) < 10):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_up_count += 1

            # Draw rectangle and label using cv2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"{classname[cls]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)  # Alternative way to draw a rectangle (commented out)
        
    # Draw the counting line
    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255, 0, 255), 2)
    cv2.putText(frame, f"Up Count: {vehicle_up_count}", (limitsUp[0], limitsUp[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("YOLO", frame)  # Display the processed frame in a window named "YOLO"
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed
        break  # If 'q' is pressed, exit the loop

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows


