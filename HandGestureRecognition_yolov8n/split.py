import os
import shutil
import random

def create_dirs(base_dir, groups, gestures):
    for group in groups:
        for gesture in gestures:
            os.makedirs(os.path.join(base_dir, 'train', group, gesture), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'val', group, gesture), exist_ok=True)

def split_data(src_dir, dest_dir, train_ratio=0.8):
    groups = [f'0{i}' for i in range(10)]
    gestures = [f'{str(i).zfill(2)}_{name}' for i, name in enumerate(['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down'], start=1)]

    create_dirs(dest_dir, groups, gestures)

    for group in groups:
        group_path = os.path.join(src_dir, group)
        if not os.path.isdir(group_path):
            continue

        for gesture in gestures:
            gesture_path = os.path.join(group_path, gesture)
            if not os.path.isdir(gesture_path):
                print(f"Directory not found: {gesture_path}")
                continue

            files = [f for f in os.listdir(gesture_path) if f.endswith('.png')]
            if not files:
                print(f"No files found for: {gesture_path}")
                continue

            random.shuffle(files)
            split_point = int(len(files) * train_ratio)

            train_files = files[:split_point]
            val_files = files[split_point:]

            for f in train_files:
                shutil.copy(os.path.join(gesture_path, f), os.path.join(dest_dir, 'train', group, gesture, f))
                txt_file = f.replace('.png', '.txt')
                if os.path.exists(os.path.join(gesture_path, txt_file)):
                    shutil.copy(os.path.join(gesture_path, txt_file), os.path.join(dest_dir, 'train', group, gesture, txt_file))

            for f in val_files:
                shutil.copy(os.path.join(gesture_path, f), os.path.join(dest_dir, 'val', group, gesture, f))
                txt_file = f.replace('.png', '.txt')
                if os.path.exists(os.path.join(gesture_path, txt_file)):
                    shutil.copy(os.path.join(gesture_path, txt_file), os.path.join(dest_dir, 'val', group, gesture, txt_file))

src_dir = r'D:\Projects\HandGestureRecognition\leapGestRecog'
dest_dir = r'D:\Projects\HandGestureRecognition\dataset'
split_data(src_dir, dest_dir)
