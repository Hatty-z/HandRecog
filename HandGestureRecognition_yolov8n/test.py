from ultralytics import YOLO

def predict_image(model_path, image_path):
    model = YOLO(model_path)

    results = model.predict(source=image_path, save=False)
    print(results)

if __name__ == "__main__":
    model_path = r'D:\Projects\HandGestureRecognition\runs\detect\yolov8n_hand_gesture\weights\best.pt'  # 替换为你实际保存的模型路径
    image_path = r'D:\Projects\HandGestureRecognition\testings\palm.jpg'  # 替换为你要测试的图片路径
    predict_image(model_path, image_path)
