import pickle
import numpy as np
from PIL import Image
import io
import onnx
from optimizer import optimize_fp_model
from calibrator import Calibrator
from evaluator import *
from torchvision import transforms

'''
# 将字节数据转换为 NumPy 数组
def bytes_to_numpy(byte_data):
    image = Image.open(io.BytesIO(byte_data))
    image = image.convert('L')  
    image = image.resize((96, 96))  
    image_np = np.array(image, dtype=np.float32)
    return image_np

def preprocess(image):
    image = image.convert('L')
    image = image.resize((96, 96))
    image_np = np.array(image, dtype=np.float32)
    image_np = (image_np / 255.0 - 0.0979) / 0.1991
    # 转换为模型输入格式 (1, 96, 96)
    image_np = np.expand_dims(image_np, axis=0)  # 从 (96, 96) 转换为 (1, 96, 96)
    return image_np
'''

def preprocess(byte_data=None, image=None):
    if byte_data is not None:
        image = Image.open(io.BytesIO(byte_data))
    elif image is not None:
        image = image.convert('L')  
    else:
        raise ValueError("Either byte_data or image must be provided.")
    
    image = image.resize((96, 96))
    image_np = np.array(image, dtype=np.float32)
    image_np = (image_np / 255.0 - 0.0979) / 0.1991
    
    image_np = np.expand_dims(image_np, axis=(0, 1))  # 从 (96, 96) 转换为 (1, 1, 96, 96)
    return image_np

# 加载，优化ONNX 模型
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)
optimized_model_path = optimize_fp_model(onnx_model_path)
print(f"Optimized model saved at: {optimized_model_path}")

# 加载校准数据集,并对数据进行预处理
with open('X_cal.pkl', 'rb') as f:
    test_images = pickle.load(f)
    # 将字节数据转换为 NumPy 数组，并进行预处理
    #calib_dataset = np.array([preprocess(Image.open(io.BytesIO(img))) for img in test_images[0:1800:20]], dtype=np.float32)
    calib_dataset = np.array([preprocess(byte_data=img) for img in test_images[0:1800:20]], dtype=np.float32)

with open('y_cal.pkl', 'rb') as f:
    test_labels = pickle.load(f)

print("Shape of calib_dataset:", calib_dataset.shape)
if calib_dataset.shape[1:] != (1, 96, 96):
    print("Adjusting shape of calib_dataset...")
    # 如果 calib_dataset 的形状为 (num_samples, 1, 1, 96, 96)，则需要调整为 (num_samples, 1, 96, 96)
    if len(calib_dataset.shape) == 5:
        calib_dataset = calib_dataset.squeeze(axis=2)  # 结果将是 (num_samples, 1, 96, 96)
    else:
        raise ValueError("Unexpected shape for calib_dataset.")
print("Adjusted shape of calib_dataset:", calib_dataset.shape)

#加载优化后的onnx模型
model_proto = onnx.load(optimized_model_path)

calib = Calibrator('int16', 'per-tensor', 'minmax')

calib.set_providers(['CPUExecutionProvider'])

# 生成量化参数
pickle_file_path = 'handrecognition_calib.pickle'
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)

# 生成系数文件
calib.export_coefficient_to_cpp(model_proto, pickle_file_path, 'esp32s3', '.', 'handrecognition_coefficient', True)
