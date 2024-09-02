import onnx
import numpy as np

def convert_to_espdl(onnx_model_path, espdl_model_path):
    model = onnx.load(onnx_model_path)
    
    # 解析ONNX模型并将其转换为ESP-DL格式
    with open(espdl_model_path, "wb") as f:
        f.write(b'ESP-DL')
        for initializer in model.graph.initializer:
            data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            f.write(data.tobytes())


convert_to_espdl(r"D:\Projects\HandGestureRecognition\best.onnx", r"D:\Projects\HandGestureRecognition\handrecognition.espdl")
