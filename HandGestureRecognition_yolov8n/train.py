from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.to('cuda')

    data_path = 'data.yaml'  
    epochs = 30
    batch_size = 4  
    img_size = 320  

    model.train(data=data_path, epochs=epochs, imgsz=img_size, batch=batch_size, name='yolov8n_hand_gesture',flipud=0,fliplr=0,patience=5)
    metrics = model.val(data=data_path, imgsz=img_size, batch=batch_size)
    print(f"Validation results: {metrics}")

if __name__ == "__main__":
    main()
