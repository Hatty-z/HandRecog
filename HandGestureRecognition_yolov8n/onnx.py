import torch
import torch.onnx
from ultralytics import YOLO

Load_Path=r"D:\Projects\HandGestureRecognition\runs\detect\yolov8n_hand_gesture\weights\best.pt"
Save_Path=r"D:\Projects\HandGestureRecognition\best.onnx"

model=YOLO('yolov8n.pt')
model.load(Load_Path)

input_sample=torch.randn(1,3,320,320)

torch.onnx.export(model.model,input_sample,Save_Path,
                  verbose=True,input_names=['input'],output_names=['output'])
print("Model converted to ONNX format and saved to", Save_Path)