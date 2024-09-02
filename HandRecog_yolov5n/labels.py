import os
import cv2

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

data_path = r'D:\Projects\HandRecog\leapGestRecog'

def get_bounding_box(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    width, height = img.shape[1], img.shape[0]  # 使用图像的实际尺寸

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height

        return x_center, y_center, w_norm, h_norm

    return None, None, None, None

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
                img = cv2.imread(img_path)

                if img is None:
                    continue

                if img.shape[:2] != (640, 240):
                    img = cv2.resize(img, (640, 240))

                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x_center, y_center, w_norm, h_norm = get_bounding_box(img) 

                if x_center is not None:
                    label_file = img_path.replace('.png', '.txt')
                    with open(label_file, 'w') as f:
                        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
