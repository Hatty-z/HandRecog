import onnx
import onnxruntime as ort
from convert_onnx import dummy_input

onnx_path = 'model.onnx'

# 加载并检查 ONNX 模型
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# 使用 ONNX Runtime 运行测试推理
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 将 dummy_input 移到 CPU 并转换为 NumPy 数组
dummy_input_cpu = dummy_input.cpu().numpy()

# 使用相同的 dummy input 进行推理测试
result = session.run([output_name], {input_name: dummy_input_cpu})
print("ONNX 模型推理结果: ", result)
