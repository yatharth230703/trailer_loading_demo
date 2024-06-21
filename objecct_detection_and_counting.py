import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\runs\detect\train7\weights\best.pt")  # Make sure you have the correct model file

# Function to detect and count objects on a live camera feed
def detect_and_count_objects_on_camera():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Get the model predictions
        results = model(frame)

        # Create a dictionary to count objects
        object_counts = {}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                confidence = box.conf.item()  # Convert to scalar
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to list and then to int

                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

                # Draw bounding box and label on the image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the object counts on the frame
        y_offset = 30
        for obj, count in object_counts.items():
            count_text = f"{obj}: {count}"
            cv2.putText(frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            y_offset += 30

        # Print object counts
        for obj, count in object_counts.items():
            print(f"{obj}: {count}")

        # Display the resulting frame
        cv2.imshow('Live Object Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live camera detection
detect_and_count_objects_on_camera()
