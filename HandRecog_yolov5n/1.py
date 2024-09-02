import onnx
import yaml
import os
from PIL import Image
import torch
import numpy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from optimizer import optimize_fp_model
from calibrator import Calibrator

class HandGestureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# 数据预处理
def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_calibration_data(data_config, batch_size, img_size):
    calibration_images_dir = data_config['calibration']
    transform = get_transform(img_size)
    dataset = HandGestureDataset(calibration_images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)
    return dataloader

def remove_unsupported_nodes(model):
    nodes_to_remove = []
    unsupported_ops = ['Constant', 'Split', 'Pow']

    for node in model.graph.node:
        if node.op_type in unsupported_ops:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    return model

# 执行静态量化
def perform_static_quantization(optimized_model_path, calibration_dataloader):
    model_proto = onnx.load(optimized_model_path)
    
    # 修剪模型
    model_proto = remove_unsupported_nodes(model_proto)

    calib = Calibrator('int16', 'per-tensor', 'minmax')
    calib.set_providers(['CPUExecutionProvider'])

    pickle_file_path = 'handrecog_calib.pickle'

    print("Generating the quantization table...")

    calib_dataset = []
    for images in calibration_dataloader:
        images = images.to('cuda').cpu().numpy()
        calib_dataset.append(images)
        torch.cuda.empty_cache()
        
    calib_dataset = numpy.concatenate(calib_dataset, axis=0)
    calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
    
    quantized_model_path = optimized_model_path.replace('.onnx', '_int16.onnx')
    onnx.save(model_proto, quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")
    return quantized_model_path

# 生成系数文件
def export_coefficients(quantized_model_path, calib_pickle_path):
    model_proto = onnx.load(quantized_model_path)
    calib = Calibrator('int16', 'per-tensor', 'minmax')
    print("Converting coefficient to int16 per-tensor quantization for esp32s3")
    calib.export_coefficient_to_cpp(
        model_proto,
        calib_pickle_path,
        'esp32s3',
        '.',  
        'hand_gesture_coefficient',
        True
    )
    print("Exporting finished, the output files are: ./hand_gesture_coefficient.cpp, ./hand_gesture_coefficient.hpp")
    
    # 打印模型信息
    print("\nQuantized model info:")
    for node in model_proto.graph.node:
        if node.op_type in ["Reshape", "Gemm"]:
            print(f"{node.op_type} layer name: {node.name}, output_exponent: -15")

def main():
    data_config_path = 'D:/Projects/HandRecog/dataset/data.yaml'
    data_config = load_yaml(data_config_path)
    
    model_path = 'D:/Projects/HandRecog/yolov5_training/weights/model.onnx'
    batch_size = 1
    img_size = 320
    
    # 优化模型
    optimized_model_path = optimize_fp_model(model_path)
    
    # 准备校准数据
    calibration_dataloader = prepare_calibration_data(data_config, batch_size, img_size)
    
    # 静态量化模型
    quantized_model_path = perform_static_quantization(optimized_model_path, calibration_dataloader)
    
    # 生成系数文件
    export_coefficients(quantized_model_path, 'handrecog_calib.pickle')

if __name__ == '__main__':
    main()
