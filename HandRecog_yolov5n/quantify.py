import onnx
import onnxoptimizer
import pickle
from optimizer import optimize_fp_model
from calibrator import Calibrator
from evaluator import Evaluator
import glob
import cv2
import numpy as np
import os

# 1. Load the ONNX model
onnx_model_path = r'D:\Projects\HandRecog\yolov5_training\weights\model.onnx'
onnx_model = onnx.load(onnx_model_path)

# 2. Optimize the ONNX model
# Use onnxoptimizer for model optimization
passes = ['fuse_consecutive_transposes', 'fuse_add_bias_into_conv', 'eliminate_identity']
optimized_model_proto = onnxoptimizer.optimize(onnx_model, passes)
optimized_model_path = 'optimized_model.onnx'
onnx.save(optimized_model_proto, optimized_model_path)

# 3. Load calibration dataset
def load_calibration_data(image_folder, label_folder):
    images = []
    labels = []

    image_files = glob.glob(f'{image_folder}/**/*.png', recursive=True)

    for img_file in image_files:
        image = cv2.imread(img_file)
        if image is None:
            continue

        if image.shape[:2] != (640, 240):
            image = cv2.resize(image, (640, 240))

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)

        label_file = img_file.replace('.png', '.txt').replace(image_folder, label_folder)
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                # Read the label file; we assume the first item is the class ID
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    labels.append(class_id)

    images = np.array(images)
    labels = np.array(labels)

    # Split calibration dataset
    calib_dataset = images[0:1800:20]  # Adjust based on actual requirements
    pickle_file_path = 'handrecognition_calib.pickle'

    with open(pickle_file_path, 'wb') as f:
        pickle.dump((calib_dataset, labels), f)

image_folder = 'D:/Projects/HandRecog/dataset/cal/images'
label_folder = 'D:/Projects/HandRecog/dataset/cal/labels'
load_calibration_data(image_folder, label_folder)

# 4. Calibration
model_proto = onnx.load(optimized_model_path)
print('Generating the quantization table:')

calib = Calibrator('int8', 'per-tensor', 'minmax')
calib.set_providers(['CPUExecutionProvider'])

# Load data from pickle file
pickle_file_path = 'handrecognition_calib.pickle'
with open(pickle_file_path, 'rb') as f:
    calib_dataset, _ = pickle.load(f)

# Generate quantization table
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)

# Export coefficient file for ESP32-S3
calib.export_coefficient_to_cpp(model_proto, pickle_file_path, 'esp32s3', '.', 'handrecognition_coefficient', True)
