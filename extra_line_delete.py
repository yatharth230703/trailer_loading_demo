import os

# Directory containing the label files
label_dir = r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolo_finetune_data\labels\val'

# Ensure the label directory exists
if not os.path.exists(label_dir):
    print(f"Directory {label_dir} does not exist.")
    exit(1)

# Process each label file
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        label_path = os.path.join(label_dir, label_file)
        
        # Read the contents of the file
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        if lines:
            # Keep only the first line
            first_line = lines[0].strip() + '\n'
            
            # Write back only the first line to the file
            with open(label_path, 'w') as file:
                file.write(first_line)
                
            print(f"Processed {label_file}: Retained only the first line")

print("All files processed.")
