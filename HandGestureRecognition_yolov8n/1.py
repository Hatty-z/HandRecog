from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.to('cuda')

    # Train the model for 1 epoch and print the returned metrics
    train_metrics = model.train(data='data.yaml', epochs=1, imgsz=320, batch=4, name='yolov8n_hand_gesture', amp=True)
    print("Train Metrics:")
    print(train_metrics)
    
    # Validate the model and print the returned metrics
    val_metrics = model.val()
    print("Validation Metrics:")
    print(val_metrics)

if __name__ == "__main__":
    main()
