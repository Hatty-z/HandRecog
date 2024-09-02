import os
import shutil
from sklearn.model_selection import train_test_split

data_path = 'leapGestRecog'
#划分比例8：2
train_ratio = 0.8
test_ratio = 0.2

train_data_path = 'dataset/train'
test_data_path = 'dataset/test'

os.makedirs(train_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)

def split_data(class_dir, dest_train_dir, dest_test_dir):
    files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    labels = [f.replace('.png', '.txt') for f in files]
    
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)
    
    for file in train_files:
        #原图片，标签文件复制到目标文件位置
        src_img = os.path.join(class_dir, file)
        src_label = os.path.join(class_dir, file.replace('.png', '.txt'))
        
        dst_img = os.path.join(dest_train_dir, file)
        dst_label = os.path.join(dest_train_dir, file.replace('.png', '.txt'))
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
    
    for file in test_files:
        src_img = os.path.join(class_dir, file)
        src_label = os.path.join(class_dir, file.replace('.png', '.txt'))
        
        dst_img = os.path.join(dest_test_dir, file)
        dst_label = os.path.join(dest_test_dir, file.replace('.png', '.txt'))
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)

for group in os.listdir(data_path):
    group_path = os.path.join(data_path, group)
    if not os.path.isdir(group_path):
        continue

    for gesture in os.listdir(group_path):
        gesture_path = os.path.join(group_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        dest_train_dir = os.path.join(train_data_path, group, gesture)
        dest_test_dir = os.path.join(test_data_path, group, gesture)
        os.makedirs(dest_train_dir, exist_ok=True)
        os.makedirs(dest_test_dir, exist_ok=True)

        #每一组的每种手势都按照比例划分训练集和测试集
        split_data(gesture_path, dest_train_dir, dest_test_dir)

print("数据集划分完成！")
