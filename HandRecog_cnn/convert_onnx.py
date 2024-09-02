import torch
from model import HandGestureCNN 

model_path = "best_model.pth" 
onnx_path = "model.onnx"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = HandGestureCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 创建一个虚拟输入张量，大小与模型期望的输入相匹配
dummy_input = torch.randn(1, 1, 96, 96).to(device)  # (batch_size, channels, height, width)

# 导出为 ONNX 格式
torch.onnx.export(
    model,               
    dummy_input,         
    onnx_path,           
    export_params=True,  # 保存模型参数
    opset_version=11,    #ONNX操作符集版本号
    do_constant_folding=True,  # 执行常量折叠优化
    input_names=['input'],   
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态批次大小
)

print(f"Model has been successfully converted to ONNX format and saved at {onnx_path}")
