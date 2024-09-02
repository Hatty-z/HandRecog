import torch
import cv2
import numpy as np
from torchvision import transforms
from model import HandGestureCNN 

# 配置
image_path = "/root/autodl-tmp/HandRecog_cnn/dataset/train/08/06_index/frame_08_06_0035.png"  # 替换为您的测试图片路径
model_path = "best_model.pth"  # 替换为您的模型权重文件路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义图像预处理函数
def preprocess_image(image_path):
    # 读取图像，确保是灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 调整大小为 (96, 96)
    img = cv2.resize(img, (96, 96))
    
    # 转换为 PyTorch 张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.09787706561797896], std=[0.19911180262177733])  # 使用之前计算的均值和标准差
    ])
    
    img_tensor = transform(img)
    
    # 添加一个批次维度
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# 加载模型
model = HandGestureCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 预处理图像
input_tensor = preprocess_image(image_path).to(device)

# 进行预测
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

# 映射类别索引到手势名称
class_names = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']  # 替换为您的实际类别名称
predicted_class = class_names[prediction]

print(f"Predicted hand gesture: {predicted_class}")
