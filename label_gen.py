import os
from ultralytics import YOLO
import shutil

# Load a pre-trained YOLOv8 model
model = YOLO(r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolov8x.pt')

# Directories
image_dir = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\chair_dataset'
output_label_dir = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\chair_label_x'

# Ensure the output directory exists
os.makedirs(output_label_dir, exist_ok=True)

# Variable to keep track of the previous detection
previous_label_file = None

# Process each image
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # Display the filename
        print(f"Processing: {image_file}")

        # Load image
        image_path = os.path.join(image_dir, image_file)
        
        # Perform detection
        results = model.predict(image_path, save=False)

        # Prepare the label file
        label_file = os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt')

        if results and len(results[0].boxes) > 0:
            with open(label_file, 'w') as f:
                for bbox in results[0].boxes:
                    x_center, y_center, width, height = bbox.xywh[0].tolist()
                    x_center /= results[0].orig_shape[1]
                    y_center /= results[0].orig_shape[0]
                    width /= results[0].orig_shape[1]
                    height /= results[0].orig_shape[0]
                    class_id = int(bbox.cls[0])
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            previous_label_file = label_file
        else:
            # If no detections, copy the previous detection
            if previous_label_file:
                shutil.copy(previous_label_file, label_file)
                print(f"No detections in {image_file}. Copied from previous detection: {previous_label_file}")

print("Processing completed.")


