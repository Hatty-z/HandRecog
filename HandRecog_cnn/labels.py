import os

#手势与标签的映射
class_mapping = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9,
}

data_path = 'leapGestRecog'

#遍历图片，根据文件夹名称对应手势种类，标注图片
for group in os.listdir(data_path):
    group_path = os.path.join(data_path, group)
    if not os.path.isdir(group_path):
        continue

    for gesture in os.listdir(group_path):
        gesture_path = os.path.join(group_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        if gesture not in class_mapping:
            continue
        class_id = class_mapping[gesture]

        for img_file in os.listdir(gesture_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(gesture_path, img_file)
                label_file = img_path.replace('.png', '.txt')
                
                with open(label_file, 'w') as f:
                    f.write(f"{class_id}\n")

print("标签文件更新完成。")