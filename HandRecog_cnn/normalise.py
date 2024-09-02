from torchvision import transforms
from PIL import Image
import os
import numpy as np

def compute_mean_std(root_dir):
    image_paths = []
    
    for group in os.listdir(root_dir):
        group_dir = os.path.join(root_dir, group)
        if os.path.isdir(group_dir):
            for gesture in os.listdir(group_dir):
                gesture_dir = os.path.join(group_dir, gesture)
                if os.path.isdir(gesture_dir):
                    for file_name in os.listdir(gesture_dir):
                        if file_name.endswith('.png'):  # 仅处理PNG格式的图像
                            img_path = os.path.join(gesture_dir, file_name)
                            image_paths.append(img_path)
    
    means, stds = [], []
    
    for img_path in image_paths:
        image = Image.open(img_path).convert('L')
        np_image = np.array(image) / 255.0  # 归一化到 [0, 1] 范围
        means.append(np_image.mean())
        stds.append(np_image.std())
    
    mean = np.mean(means)
    std = np.mean(stds)
    
    return mean, std

# 计算训练集的均值和标准差
train_mean, train_std = compute_mean_std('/root/autodl-tmp/HandRecog_cnn/dataset/train')

print(f"Mean: {train_mean}, Std: {train_std}")
