import os
import pickle
import shutil
from sklearn.model_selection import train_test_split

train_data_path = 'dataset/train'
test_data_path = 'dataset/test'
calib_data_path = 'dataset/calib'

os.makedirs(calib_data_path, exist_ok=True)

def get_files_from_dir(data_path):
    files = []
    for group in os.listdir(data_path):
        group_path = os.path.join(data_path, group)
        if not os.path.isdir(group_path):
            continue

        for gesture in os.listdir(group_path):
            gesture_path = os.path.join(group_path, gesture)
            if not os.path.isdir(gesture_path):
                continue

            for file in os.listdir(gesture_path):
                if file.endswith('.png'):
                    img_path = os.path.join(gesture_path, file)
                    label_path = os.path.join(gesture_path, file.replace('.png', '.txt'))
                    #将每个图像文件路径和对应的标签文件路径作为一个元组 (img_path, label_path)，并将这个元组添加到 files 列表中
                    files.append((img_path, label_path))
    return files

train_files = get_files_from_dir(train_data_path)
test_files = get_files_from_dir(test_data_path)

# 设定提取比例
ts = 0.3  # 从训练集中和测试集中各提取 30% 的数据作为校准集的一部分

_, calib_train_files = train_test_split(train_files, test_size=ts, random_state=42)
_, calib_test_files = train_test_split(test_files, test_size=ts, random_state=42)

# 合并校准数据集
calib_files = calib_train_files + calib_test_files

# 将校准数据集复制到新的目录
for img_path, label_path in calib_files:
    calib_img_path = img_path.replace('dataset/train', 'dataset/calib').replace('dataset/test', 'dataset/calib')
    calib_label_path = label_path.replace('dataset/train', 'dataset/calib').replace('dataset/test', 'dataset/calib')

    os.makedirs(os.path.dirname(calib_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(calib_label_path), exist_ok=True)

    shutil.copy(img_path, calib_img_path)
    shutil.copy(label_path, calib_label_path)

print("校准数据集提取完成！")

# 保存校准数据集为.pkl文件
calib_images = []
calib_labels = []

for img_path, label_path in calib_files:
    #使用二进制模式 ('rb') 打开文件并读取其内容，将其作为字节流数据添加到 calib_images 列表中。
    with open(img_path, 'rb') as f:
        calib_images.append(f.read())  #加载图片

    with open(label_path, 'r') as f:
        calib_labels.append(f.read())  

# 以二进制写模式 ('wb') 打开（或创建）一个文件
with open('X_cal.pkl', 'wb') as f:
    #将 calib_images 列表序列化并写入文件 f
    pickle.dump(calib_images, f)

with open('y_cal.pkl', 'wb') as f:
    pickle.dump(calib_labels, f)

print("校准数据集已保存为.pkl文件！")
