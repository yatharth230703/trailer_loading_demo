from ultralytics import YOLO

def main():
    # Load a YOLOv8 model (you can also specify the model path)
    model = YOLO(r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolov8n.pt')  

    # Train the model with a different image size
    model.train(data=r'C:\Users\Yatharth\Desktop\desktop1\AI\Sunic\how_many_Are_kept_on_trailer\yolo_train.yaml', epochs=50, imgsz=640, batch=16)

if __name__ == '__main__':
    main()
