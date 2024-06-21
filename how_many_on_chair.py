from ultralytics import YOLO
import cv2

# Paths to your models
chair_model_path = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\runs\detect\train7\weights\best.pt'
object_model_path = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolov8n.pt'  # Specify the path to the object detection model

# Load the YOLO models
chair_model = YOLO(chair_model_path)
object_model = YOLO(object_model_path)

# Open a connection to the webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or replace with the video source path

if not video_capture.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Detect chairs
    chair_results = chair_model(frame)
    
    # Iterate through detected chairs
    for chair_result in chair_results:
        for chair_box in chair_result.boxes:
            # Extract chair bounding box coordinates
            x1, y1, x2, y2 = map(int, chair_box.xyxy[0])
            
            # Crop the chair from the frame
            chair_crop = frame[y1:y2, x1:x2]
            
            # Detect objects on the chair
            object_results = object_model(chair_crop)
            
            # Count detected objects
            object_count = len(object_results[0].boxes) if object_results else 0
            
            # Draw the chair bounding box and object count on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Objects: {object_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
