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
        
        if not lines:
            continue 
        
        first_line = lines[0].strip().split()
        if first_line[0] != '0':
           
            first_line[0] = '0'
            lines[0] = ' '.join(first_line) + '\n'
            
            with open(label_path, 'w') as file:
                file.writelines(lines)
                
            print(f"Modified {label_file}: Changed first number to 0")

print("Verification and modification completed.")
