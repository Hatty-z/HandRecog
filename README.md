# 模型开发
1. **数据**
- 收集
- 划分数据集（训练集，验证集，测试集）
- 预处理
> 提高数据质量，模型训练效果
2. **模型的选择与设计**
3. **模型训练，验证与调优**
- 使用训练集进行模型训练
- 使用验证集进行模型性能评估
- 调整超参数（学习率，批次大小，迭代次数）
> 确保模型具有良好的性能与泛化能力
4. **模型评估**
使用测试集进行模型性能评估
***评估指标***（准确率，召回率，F1_score等）
5. **模型优化**
- 模型剪枝：减少模型参数，去除冗余节点，提高推理速度。
- 模型量化：将模型从浮点数精度降低到定点数精度（如FP32到INT8），减少计算量和内存占用。
量化时遇到两种量化方法选择
查阅资料：
两种量化方式的比较
(A) 动态量化 (Dynamic Quantization)
动态量化在推理过程中对激活值（如权重和中间结果）进行量化。这意味着模型在推理时会动态地将某些浮点运算转换为整数运算，以提高推理效率。此方法不需要校准数据集，适合一些模型在部署时简化量化过程。
- 优点:
  - 不需要校准数据集，过程较为简单。
  - 可以立即用于大部分推理任务。
  - 通常会保留浮点精度，适用于大部分任务。
- 缺点:
  - 对硬件加速器的支持不如静态量化好。
  - 在一些情况下，推理速度的提升有限。
(B) 静态量化 (Static Quantization)
静态量化需要在推理之前对整个模型进行量化。它通常使用校准数据集来估计模型的激活范围，并将其转化为整数格式。这种方法可以显著降低模型的大小和计算复杂度，并且在硬件加速器（如ESP32-S3）上表现更好。
- 优点:
  - 大幅减少模型的大小和计算成本。
  - 在硬件加速器上的性能优化效果更明显。
  - 可用于部署在资源受限的设备上，如微控制器。
- 缺点:
  - 需要一个校准数据集，步骤较为复杂。
  - 可能会导致精度下降，尤其是在模型对量化敏感的情况下。
1. 如何选择
- 如果你需要一个较为简单的量化过程，并且对性能要求不高，可以选择动态量化。
- 如果你希望最大化模型在硬件上的性能，且有校准数据集，那么静态量化可能是更好的选择。
- 蒸馏学习：使用一个大模型（教师模型）指导小模型（学生模型）学习，提高小模型的性能。
减少模型的计算量与内存占用，提高推理速度与部署效率
6. 模型部署
选择合适的推理框架（TensorFlow Lite, ONNX Runtime等）
查阅TensorFlow Lite  & ONNX Runtime
平台和兼容性：
- TensorFlow Lite：主要设计用于移动和嵌入式设备上，如Android和iOS手机。它对TensorFlow生态系统有很好的支持。
- ONNX Runtime：支持多种平台，包括Windows、Linux、MacOS以及移动设备。它可以执行用多种框架训练的模型，只要这些模型被转换成ONNX格式，如PyTorch、TensorFlow、Scikit-Learn等。
模型转换和支持：
- TensorFlow Lite：需要将TensorFlow模型转换成TFLite格式。这一过程可能涉及到功能的简化或修改，因为TFLite不支持TensorFlow的全部操作。
- ONNX Runtime：可以加载ONNX格式的模型，这是一个开放的模型格式，支持多种深度学习框架。如果你的模型是用PyTorch、MXNet等其他框架训练的，ONNX Runtime可能是更好的选择。
性能和优化：
- TensorFlow Lite：提供了多种优化选项，包括量化和使用硬件加速（如GPU和TPU）。
- ONNX Runtime：也提供优化和硬件加速支持，包括使用NVIDIA的TensorRT、Intel的DNNL等。它的性能优化通常在服务器和云环境中表现更为突出。
7.  监控与维护
- 实时监控：监控模型在实际使用中的表现，及时发现和解决问题。
- 定期更新：根据新数据和新需求，定期更新和重新训练模型，保持模型性能。
手势识别模型
1. 数据
1.1 标注
用于识别原始数据（图片、文本文件、视频等）并添加一个或多个有意义的信息标签以提供上下文，从而使机器学习模型能够从中学习
每张图片生成对应标签文件
标签文件仅含有一个表示手势种类的标签
多个标签可能导致训练过程中标签文件读取错误，也可以处理只读取手势类别的标签
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
1.2 划分数据集
这里划分了训练集和测试集，校准集从训练集和测试集中随机抽取
划分训练集和测试集
import os
import shutil
from sklearn.model_selection import train_test_split

data_path = 'leapGestRecog'
#划分比例8：2
train_ratio = 0.8
test_ratio = 0.2

train_data_path = 'dataset/train'
test_data_path = 'dataset/test'

os.makedirs(train_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)

def split_data(class_dir, dest_train_dir, dest_test_dir):
    files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    labels = [f.replace('.png', '.txt') for f in files]
    
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)
    
    for file in train_files:
        #原图片，标签文件复制到目标文件位置
        src_img = os.path.join(class_dir, file)
        src_label = os.path.join(class_dir, file.replace('.png', '.txt'))
        
        dst_img = os.path.join(dest_train_dir, file)
        dst_label = os.path.join(dest_train_dir, file.replace('.png', '.txt'))
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
    
    for file in test_files:
        src_img = os.path.join(class_dir, file)
        src_label = os.path.join(class_dir, file.replace('.png', '.txt'))
        
        dst_img = os.path.join(dest_test_dir, file)
        dst_label = os.path.join(dest_test_dir, file.replace('.png', '.txt'))
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)

for group in os.listdir(data_path):
    group_path = os.path.join(data_path, group)
    if not os.path.isdir(group_path):
        continue

    for gesture in os.listdir(group_path):
        gesture_path = os.path.join(group_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        dest_train_dir = os.path.join(train_data_path, group, gesture)
        dest_test_dir = os.path.join(test_data_path, group, gesture)
        os.makedirs(dest_train_dir, exist_ok=True)
        os.makedirs(dest_test_dir, exist_ok=True)

        #每一组的每种手势都按照比例划分训练集和测试集
        split_data(gesture_path, dest_train_dir, dest_test_dir)

print("数据集划分完成！")

---
划分校准集并存储为.pkl文件
由于后续量化参考教程划分校准数据集，保存为.pkl，其实可以直接划分出校准数据集，保存到dataset/image目录，后续对校准数据集的预处理也较为方便
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

---
查阅.pkl文件
序列化数据：.pkl 文件使用 Python 的 pickle 模块保存，pickle 可以将 Python 对象序列化为字节流，这样可以方便地将数据对象（如列表、字典、NumPy 数组等）存储到文件中。
快速加载：相较于原始的文件格式（如文本文件或图像文件），pickle 可以更快地加载数据，因为它是直接将二进制数据加载回内存，而不需要进行文本解析或图像解码。
保存复杂数据结构：pickle 可以直接保存复杂的数据结构，例如包含多种类型的 Python 对象的列表或字典，而不需要进行特殊的转换。
便于后续使用：机器学习和数据科学中的很多工作流需要频繁加载和保存数据集，使用 .pkl 文件格式可以更高效地保存数据对象的状态，并在以后使用时方便地加载。
1.3 数据预处理（包含在模型训练脚本中）

2. 模型选择与设计
查阅信息
1. 几个手势识别模型
以下是几种常见的手势识别模型：
1. 传统方法
- HOG + SVM：Histogram of Oriented Gradients (HOG) 特征提取方法与支持向量机（SVM）分类器结合。
- Haar-like 特征 + AdaBoost：使用 Haar-like 特征进行特征提取，然后通过 AdaBoost 算法进行分类。
2. 深度学习方法
- 卷积神经网络 (CNN)：例如LeNet, AlexNet, VGG, ResNet等，用于特征提取和分类。
- 区域卷积神经网络 (R-CNN)：如 Faster R-CNN、Mask R-CNN，用于目标检测。
- YOLO (You Only Look Once)：YOLOv3, YOLOv4, YOLOv5, YOLOv8等，用于实时目标检测和识别。
- SSD (Single Shot MultiBox Detector)：用于实时目标检测。
- MobileNet：轻量级卷积神经网络，适用于移动设备。
- OpenPose：用于人体关键点检测的深度学习模型。
2. 适合部署在硬件上的模型
在硬件上部署模型时，需要考虑模型的计算复杂度、内存需求以及推理速度。以下是几种适合部署在硬件上的轻量级模型：
- MobileNet：特别设计用于移动和嵌入式设备，计算效率高。
- YOLOv5/YOLOv8：其中的轻量级版本（如YOLOv5s/YOLOv8s）在嵌入式设备上表现良好。
- Tiny-YOLO：YOLO的精简版本，适合资源受限的设备。
- SqueezeNet：一种小型化的CNN，减少参数量和计算量。
- EfficientNet-Lite：EfficientNet的轻量级版本，专为移动设备优化。
3. 部署到硬件上的难点
- 计算资源有限：嵌入式设备通常计算能力和内存有限，需要优化模型的大小和计算复杂度。
- 功耗限制：移动设备的电池寿命是一个重要考虑因素，高效的模型设计和计算方法是必要的。
- 实时性要求：手势识别通常需要实时处理，模型需要具备快速推理能力。
- 硬件兼容性：不同的硬件平台（如NVIDIA Jetson, Raspberry Pi, FPGA, DSP等）对模型优化和部署有不同的要求。
- 软件支持：需要合适的软件框架和工具链（如TensorFlow Lite, ONNX Runtime, OpenVINO等）来支持模型部署和推理。
部署优化技巧
- 模型剪枝和量化：减少模型参数和计算量，提升推理速度。
- 硬件加速：利用硬件的加速能力，如GPU, NPU, TPU等。
- 混合精度计算：使用FP16或INT8计算代替FP32，提高计算效率。
- 输入分辨率调整：根据应用场景适当降低输入分辨率，减少计算负荷
cnn模型
HandGestureCNN 类
[] 构造初始化模型架构和优化器，定义损失函数
def __init__(self,learning_rate=0.001):
        super(HandGestureCNN,self).__init__()
        self.model1=Sequential(
            Conv2d(1,32,5),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.2),
            Conv2d(32,64,3),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.2),
            Conv2d(64,64,3),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(64*10*10, 128), 
            ReLU(),
            Linear(128,10),
            Softmax(dim=1)
        )
        self.loss_fn=CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate)
        
  def forward(self, x):
        x = self.model1(x)
        return x
[] 保存，加载模型检查点（可选）
刚开始进行模型训练，想法是每次训练较少轮数，训练结束后保存模型检查点（包括当前epoch数、模型状态字典以及优化器状态字典），下一次训练加载模型的检查点，如果文件存在，则恢复模型和优化器的状态，并返回上次保存的epoch数加1；否则返回0。
这样就可以在前一次训练的基础上继续训练
但是 使用早停机制更为方便，并且可以防止过拟合
    def save_checkpoint(self, epoch, path='checkpoint.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path='checkpoint.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1
        return 0
EarlyStopping 类
实现早停机制
[] 初始化早停机制参数
def __init__(self, patience=5, delta=0):
    self.patience = patience
    self.delta = delta
    self.best_score = None
    self.counter = 0
    self.early_stop = False
[] 比较当前的验证损失 
val_loss 和历史最佳损失，如果当前损失优于历史最佳损失，则更新 best_score 并重置计数器 counter，同时保存当前模型状态；如果当前损失没有改进，则增加计数器 counter。如果计数器达到 patience，则设置 early_stop 标志为真，表示应该停止训练。
def __call__(self, val_loss, model):
    if self.best_score is None:
        self.best_score = val_loss
    elif val_loss < self.best_score - self.delta:
        self.best_score = val_loss
        self.counter = 0
        
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
yolov8n
yolov8n和yolov5n进行了尝试，但是在部署阶段，由于模型较为复杂，很多网络层espdl不支持
可能需要重新定义模型
但是模型部署前的训练都已完成，并且识别准确率很高
这部分的文档后续补充，代码已经上传。。期待后面的更新。。。
也其他您提出建议！！
数据标注
1. YOLO的标签文件需要每一行包含如下信息：
<class_id> <x_center> <y_center> <width> <height>
- <class_id>：类别ID，从0开始。
- <x_center>：目标边界框中心点的x坐标，相对于图片宽度的比例。
- <y_center>：目标边界框中心点的y坐标，相对于图片高度的比例。
- <width>：目标边界框的宽度，相对于图片宽度的比例。
- <height>：目标边界框的高度，相对于图片高度的比例。
2. 数据集图片已经分类到不同的文件夹中，并且你有一个类名到ID的映射（例如：01_palm -> 0, 10_down -> 9），需要编写一个脚本来生成YOLO格式的标签文件。
3. github上下载模型，安装依赖
yolov5n
cd D:/Projects/HandRecog/yolov5
python train.py --img 320 --epochs 1 --data ../dataset/data.yaml --weights yolov5n.pt --batch-size 4 --patience 5 --hyp D:/Projects/HandRecog/yolov5/data/hyps/hyp.scratch-low.yaml --project D:/Projects/HandRecog --name runs
3. 模型训练,验证与调优
3.1 准备数据集
3.1.1 自定义数据集类
不使用PyTorch的内置数据加载器（如torch.utils.data.DataLoader）
不符合标准数据集格式
[] __init__函数初始化一些参数，如读取外部数据源文件。
def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.img_labels = []
    
    for group in os.listdir(root_dir):
        group_dir = os.path.join(root_dir, group)
        if os.path.isdir(group_dir):
            for gesture in os.listdir(group_dir):
                gesture_dir = os.path.join(group_dir, gesture)
                if os.path.isdir(gesture_dir):
                    for file_name in os.listdir(gesture_dir):
                        if file_name.endswith('.png'):  
                            img_path = os.path.join(gesture_dir, file_name)
                            label_path = img_path.replace('.png', '.txt')  
                            if os.path.exists(label_path):
                                with open(label_path, 'r') as f:
                                    label = int(f.read().strip())  # 从标签文件中读取类别
                                self.img_labels.append((img_path, label))
[] __len__函数获取数据的总量。
def __len__(self):
    return len(self.img_labels)
[] __getitem__函数，对索引的数据集进行预处理
def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image, label
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0979], std=[0.1991])  
])
- 数据集图片本身为灰度图像，但是经过处理后会被转换为RGB格式
所以先统一转换为灰度图像
- 调整图像大小
- 转换为张量
将图像的像素值从 [0, 255] 范围的整数转换为 [0, 1] 范围的浮点数，并且将图像从 (H, W, C) 的形状转换为 (C, H, W) 的形状，其中 C 是颜色通道（对于灰度图像是 1，RGB 图像是 3）
- 标准化（将图像数据调整为均值为 0 和标准差为 1 的分布）
针对灰度图像的均值和标准差的计算
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def compute_mean_std(root_dir):
    image_paths = []
    
    for group in os.listdir(root_dir):
        group_dir = os.path.join(root_dir, group)
        if os.path.isdir(group_dir):
            for gesture in os.listdir(group_dir):
                gesture_dir = os.path.join(group_dir, gesture)
                if os.path.isdir(gesture_dir):
                    for file_name in os.listdir(gesture_dir):
                        if file_name.endswith('.png'):  # 仅处理PNG格式的图像
                            img_path = os.path.join(gesture_dir, file_name)
                            image_paths.append(img_path)
    
    means, stds = [], []
    
    for img_path in image_paths:
        image = Image.open(img_path).convert('L')
        np_image = np.array(image) / 255.0  # 归一化到 [0, 1] 范围
        means.append(np_image.mean())
        stds.append(np_image.std())
    
    mean = np.mean(means)
    std = np.mean(stds)
    
    return mean, std

# 计算训练集的均值和标准差
train_mean, train_std = compute_mean_std('/root/autodl-tmp/HandRecog_cnn/dataset/train')

print(f"Mean: {train_mean}, Std: {train_std}")

3.1.2 Dataloader 打乱顺序，划分批次，加载数据
train_dataset = GestureDataset('/root/autodl-tmp/HandRecog_cnn/dataset/train', transform=transform)
test_dataset = GestureDataset('/root/autodl-tmp/HandRecog_cnn/dataset/test', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

3.2 训练
实例化，加载模型，初始化参数（训练轮数，学习率，早停机制），开始训练
# 实例化，加载模型
model = HandGestureCNN().to(device)
start_epoch = model.load_checkpoint()  
additional_epochs = 50
total_train_step = start_epoch * len(train_dataloader)

# 学习率调度器
#在每个epoch结束时根据 step_size 自动调整学习率，gamma 参数控制学习率衰减的比例
scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=5, gamma=0.1)

# 初始化早停机制
early_stopping = EarlyStopping(patience=5, delta=0.01)

for epoch in range(start_epoch, start_epoch + additional_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = model.loss_fn(outputs, labels)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        total_train_step += 1

        writer.add_scalar('Training Loss', loss.item(), total_train_step)

    avg_train_loss = running_loss / len(train_dataset)  
    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')

3.3 评估
对于测试集的图片
得到模型输出，预测标签
累计总数，正确预测的数量
计算损失并累加
计算准确率，平均损失
 model.eval()
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = model.loss_fn(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(test_dataset)  

    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}')

    writer.add_scalar('Validation Accuracy', accuracy, epoch + 1)
    writer.add_scalar('Validation Loss', avg_val_loss, epoch + 1)

3.4 调优
更新学习率调度器
检查是否触发早停机制
保存模型状态
    # 学习率调度
    scheduler.step()

    # 保存最新模型
    torch.save(model.state_dict(), 'latest_model.pth')

    # 使用早停机制
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

torch.save(model.state_dict(), 'final_model.pth')
writer.close()
3.5 结果
[图片]
4. 模型转换
部署模型参考
部署示例
模型量化了解
4.1 转换为onnx格式模型
量化工具包基于开源的 AI 模型格式 ONNX 运行。其他平台训练得到的模型需要先转换为 ONNX 格式才能使用该工具包。
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

4.2 转换为esp-dl适配模型（使用量化工具包）
量化工具包文档
环境要求
- Python == 3.7
- Numba == 0.53.1
- ONNX == 1.9.0
- ONNX Runtime == 1.7.0
- ONNX Optimizer == 0.2.6
遇到问题
1. 原先模型训练的环境中（python=3.8），calibrator.pyd没办法正常加载，新的环境（python3.7）里可以加载
在原环境中导出onnx格式模型文件，在新环境进行优化，量化以及部署
2. 导出的onnx版本不兼容的问题
原环境中onnx，onnx runtime，onnx optimizer版本与新环境保持一致
3. 原环境中onnx，onnx runtime，onnx optimizer版本无法降低
解决：修改export.py对版本的检查
check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.3.0"))
check_requirements("onnx>=1.9.0")
这样就成功导出onnx格式的模型文件，并且新环境中不会有onnx版本不兼容的问题啦~

1. 模型优化
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)
optimized_model_path = optimize_fp_model(onnx_model_path)
print(f"Optimized model saved at: {optimized_model_path}")
2. 模型量化
- 加载校准数据集
with open('X_cal.pkl', 'rb') as f:
    test_images = pickle.load(f）
    calib_dataset = np.array([preprocess(byte_data=img) for img in test_images[0:1800:20]], dtype=np.float32)
    
with open('y_cal.pkl', 'rb') as f:
    test_labels = pickle.load(f)
- 对数据进行预处理（和模型训练时的预处理一致）
由于校准数据集保存为.pkl文件，需要先将字节流数据转换为numpy数组
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
遇到问题：
预处理函数 preprocess 输出了一个形状为 (1, 1, 96, 96) 的张量。如果将单个样本作为批量处理，则正常的形状应该是 (batch_size, channels, height, width)，在这个情况下即 (1, 1, 96, 96)。但是，如果 calib_dataset 是一个包含多个这样的样本的数组，那么它的形状将会是 (num_samples, 1, 1, 96, 96)，因此会得到一个秩为 5 的输入。
[图片]
增加调试代码，检查张量形状，不符合输入要求，则更改形状
[图片]

---
print("Shape of calib_dataset:", calib_dataset.shape)
if calib_dataset.shape[1:] != (1, 96, 96):
    print("Adjusting shape of calib_dataset...")
    # 如果 calib_dataset 的形状为 (num_samples, 1, 1, 96, 96)，则需要调整为 (num_samples, 1, 96, 96)
    if len(calib_dataset.shape) == 5:
        calib_dataset = calib_dataset.squeeze(axis=2)  # 结果将是 (num_samples, 1, 96, 96)
    else:
        raise ValueError("Unexpected shape for calib_dataset.")
print("Adjusted shape of calib_dataset:", calib_dataset.shape)


---
- 校准
  - 加载优化后的onnx模型
  - 实例化了一个 Calibrator 对象，负责量化校准过程
  - 设置 ONNX 运行时使用的执行提供者
  - 使用校准数据集 calib_dataset 来确定模型中每个张量的最佳量化参数，并将这些参数存储在 pickle_file_path 指定的文件中。
  - 将量化参数从 pickle_file_path 文件导出为 C++ 代码
#加载优化后的onnx模型
model_proto = onnx.load(optimized_model_path)

calib = Calibrator('int16', 'per-tensor', 'minmax')

calib.set_providers(['CPUExecutionProvider'])

# 生成量化参数
pickle_file_path = 'handrecognition_calib.pickle'
calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)

# 生成系数文件
calib.export_coefficient_to_cpp(model_proto, pickle_file_path, 'esp32s3', '.', 'handrecognition_coefficient', True)
5. 模型部署1
esp-idf安装指南
模型部署1与2思路做法基本一致，只有模型定义不太一样
- 模型部署2流程完整，但是最终的识别准确率不高
- 模型部署1出现，最终监视输出时出现5.3部署模型中的问题（尚未解决，其他你的解决方案），也可以把部署流程走一遍，但是看不到结果www
5.1 构建模型
5.1.1 
[] 从 include/layer/dl_layer_model.hpp 中的模型类派生一个新类，将层声明为成员变量
- 全连接层使用库dl_layer_fullyconnected.hpp
- 在 C++ 中，using namespace 是一种命名空间使用的声明，它允许你在不指定完整命名空间限定的情况下使用该命名空间内的标识符（如类型、函数、变量等）。
  不需要写成如下：
暂时无法在成电飞书文档外展示此内容
class HANDRECOGNITION : public Model<int16_t>
{
}；
private:
    Conv2D<int16_t> l1;
    Relu<int16_t> l2;
    MaxPool2D<int16_t> l3;
    Conv2D<int16_t> l4;
    Relu<int16_t> l5; 
    MaxPool2D<int16_t> l6;
    Conv2D<int16_t> l7;
    Relu<int16_t> l8; 
    MaxPool2D<int16_t> l9;
    Flatten<int16_t> l10;
    FullyConnected<int16_t> l11;
    Relu<int16_t> l12;
    FullyConnected<int16_t> l13;
public:
    Softmax<int16_t> l14;
[] 用构造函数初始化层
根据handrecognition_coefficient.hpp，量化表编写
[图片]
 HANDRECOGNITION() : 
    l1(Conv2D<int16_t>(-12, get_conv_0_filter(), get_conv_0_bias(), get_conv_0_activation(), PADDING_VALID, {}, 1, 1, "l1")),
    l2(Relu<int16_t>()),
    l3(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),        
    l4(Conv2D<int16_t>(-11, get_conv_3_filter(), get_conv_3_bias(), get_conv_3_activation(), PADDING_VALID, {}, 1, 1, "l4")),
    l5(Relu<int16_t>()),
    l6(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l6")),        
    l7(Conv2D<int16_t>(-9, get_conv_6_filter(), get_conv_6_bias(), get_conv_6_activation(), PADDING_VALID, {}, 1, 1, "l7")),
    l8(Relu<int16_t>()),
    l9(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l9")),        
    l10(Flatten<int16_t>()),
    l11(FullyConnected<int16_t>(-7, get_gemm_10_filter(), get_gemm_10_bias(), get_gemm_10_activation(), "l11")),
    l12(Relu<int16_t>()),
    l13(FullyConnected<int16_t>(-8, get_gemm_12_filter(), get_gemm_12_bias(), NULL, "l13")),
    l14(Softmax<int16_t>(-14))
{}
5.1.2 实现build函数
void build(Tensor<int16_t> &input)
{
    this->l1.build(input);
    this->l2.build(this->l1.get_output());
    this->l3.build(this->l2.get_output());
    this->l4.build(this->l3.get_output());
    this->l5.build(this->l4.get_output());
    this->l6.build(this->l5.get_output());
    this->l7.build(this->l6.get_output());
    this->l8.build(this->l7.get_output());
    this->l9.build(this->l8.get_output());
    this->l10.build(this->l9.get_output());
    this->l11.build(this->l10.get_output());
    this->l12.build(this->l11.get_output());
    this->l13.build(this->l12.get_output());
    this->l14.build(this->l13.get_output());       
}
5.1.3 实现call函数
void call(Tensor<int16_t> &input)
{
    this->l1.call(input);
    input.free_element();

    this->l2.call(this->l1.get_output());
    this->l1.get_output().free_element();

    this->l3.call(this->l2.get_output());
    this->l2.get_output().free_element();

    this->l4.call(this->l3.get_output());
    this->l3.get_output().free_element();

    this->l5.call(this->l4.get_output());
    this->l4.get_output().free_element();

    this->l6.call(this->l5.get_output());
    this->l5.get_output().free_element();

    this->l7.call(this->l6.get_output());
    this->l6.get_output().free_element();

    this->l8.call(this->l7.get_output());
    this->l7.get_output().free_element();

    this->l9.call(this->l8.get_output());
    this->l8.get_output().free_element();

    this->l10.call(this->l9.get_output());
    this->l9.get_output().free_element();

    this->l11.call(this->l10.get_output());
    this->l10.get_output().free_element();

    this->l12.call(this->l11.get_output());
    this->l11.get_output().free_element();

    this->l13.call(this->l12.get_output());
    this->l12.get_output().free_element();

    this->l14.call(this->l13.get_output());
    this->l13.get_output().free_element();
}
5.2 运行模型
- 创建模型对象
- 定义输入
  - 输入图像大小
 int input_height = 96;
int input_width = 96;
int input_channel = 1;
int input_exponent = -13;
  - 量化输入，归一化+定点化（生成数组已经定义）
//为输入数据分配内存
int16_t *model_input = (int16_t *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof(int16_t *));
//归一化，定点化
for(int i=0 ;i<input_height*input_width*input_channel; i++){
    model_input[i] = example_element[i];
}
  - 定义输入张量
Tensor<int16_t> input;
input.set_element((int16_t *)model_input).set_exponent(input_exponent).set_shape({96, 96, 1}).set_auto_free(false);
- 推理
  - 初始化模型。
  - 使用 Latency 对象测量模型推理的延迟（前向传播时间）。
  - 调用 model.forward(input) 进行推理
HANDRECOGNITION model;
dl::tool::Latency latency;
latency.start();
model.forward(input);
latency.end();
latency.print("HANDRECOGNITION", "forward");
  - 预测模型结果
float *score = model.l14.get_output().get_element_ptr();
    float max_score = score[0];
    int max_index = 0;
    printf("%f, ", max_score);

    for (size_t i = 1; i < 10; i++)
    {
        printf("%f, ", score[i]);
        if (score[i] > max_score)
        {
            max_score = score[i];
            max_index = i;
        }
    }

    const char *gesture_names[] = {
        "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c", "down"
    };

    printf("\nPrediction Result: %d,%s\n", max_index,gesture_names[max_index]);
5.3 部署模型
可见部署模型2部分
内存相关问题
[图片]
尝试进行了一些调试
[图片]
[图片]
[图片]
6. 模型部署2
esp-idf安装指南
6.1 构建模型
6.1.1 
[] 导入库
#pragma once
#include <stdint.h>
#include "layer/dl_layer_model.hpp"
#include "layer/dl_layer_base.hpp"
#include "layer/dl_layer_max_pool2d.hpp"
#include "layer/dl_layer_conv2d.hpp"
#include "layer/dl_layer_reshape.hpp"
#include "layer/dl_layer_softmax.hpp"
#include "layer/dl_layer_relu.hpp"
#include "handrecognition_coefficient.hpp"
[] 从 include/layer/dl_layer_model.hpp 中的模型类派生一个新类，将层声明为成员变量
1. reshape层代替flatten层
2. Conv2d层代替全连接层
否则
经过flatten层展平后，图片形状不符合二维卷积层的输入要求
class HANDRECOGNITION : public Model<int16_t>
{
}；
private:
    Conv2D<int16_t> l1;
    Relu<int16_t> l2;
    MaxPool2D<int16_t> l3;
    Conv2D<int16_t> l4;
    Relu<int16_t> l5; 
    MaxPool2D<int16_t> l6;
    Conv2D<int16_t> l7;
    Relu<int16_t> l8; 
    MaxPool2D<int16_t> l9;
    Reshape<int16_t> l10;
    Conv2D<int16_t> l11;
    Relu<int16_t> l12;
    Conv2D<int16_t> l13;
public:
    Softmax<int16_t> l14; // output layer
[] 用构造函数初始化层
根据handrecognition_coefficient.hpp，量化表编写
[图片]
HANDRECOGNITION() : 
    l1(Conv2D<int16_t>(-12, 
                       get_conv_0_filter(), 
                       get_conv_0_bias(), 
                       get_conv_0_activation(), 
                       PADDING_VALID, {}, 1, 1, "l1")),
    l2(Relu<int16_t>("l2")),
    l3(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l3")),                      
    l4(Conv2D<int16_t>(-11, 
                    get_conv_3_filter(), 
                    get_conv_3_bias(), 
                    get_conv_3_activation(), 
                    PADDING_VALID, {}, 1, 1, "l4")),
    l5(Relu<int16_t>("l5")),
    l6(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l6")),
    l7(Conv2D<int16_t>(-9, 
                    get_conv_6_filter(), 
                    get_conv_6_bias(), 
                    get_conv_6_activation(), 
                    PADDING_VALID, {}, 1, 1, "l7")),
    l8(Relu<int16_t>("l8")),
    l9(MaxPool2D<int16_t>({2, 2}, PADDING_VALID, {}, 2, 2, "l9")),
    l10(Reshape<int16_t>({1, 1, 6400}, "l10")),
    l11(Conv2D<int16_t>(-7, 
                    get_gemm_10_filter(), 
                    get_gemm_10_bias(), 
                    get_gemm_10_activation(), 
                    PADDING_VALID, {}, 1, 1, "l11")),
    l12(Relu<int16_t>("l12")),
    l13(Conv2D<int16_t>(-8, 
                    get_gemm_12_filter(), 
                    get_gemm_12_bias(), 
                    NULL, 
                    PADDING_VALID, {}, 1, 1, "l13")),
    l14(Softmax<int16_t>(-14, "l14")) 
6.1.2 实现build函数
void build(Tensor<int16_t> &input)
{
    this->l1.build(input);
    this->l2.build(this->l1.get_output());
    this->l3.build(this->l2.get_output());
    this->l4.build(this->l3.get_output());
    this->l5.build(this->l4.get_output());
    this->l6.build(this->l5.get_output());
    this->l7.build(this->l6.get_output());
    this->l8.build(this->l7.get_output());
    this->l9.build(this->l8.get_output());
    this->l10.build(this->l9.get_output());
    this->l11.build(this->l10.get_output());
    this->l12.build(this->l11.get_output());
    this->l13.build(this->l12.get_output());
    this->l14.build(this->l13.get_output());       
}
6.1.3 实现call函数
void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();

        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();

        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();

        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();

        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();

        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();

        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();

        this->l12.call(this->l11.get_output());
        this->l11.get_output().free_element();

        this->l13.call(this->l12.get_output());
        this->l12.get_output().free_element();

        this->l14.call(this->l13.get_output());
        this->l13.get_output().free_element();
    }
6.2 运行模型
- 创建模型对象
- 定义输入
  - 输入图像大小
int input_height = 96;
int input_width = 96;
int input_channel = 1;
int input_exponent = -13; 
  - 量化输入，归一化+定点化（生成数组时已经进行）
    - 生成数组脚本
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((96, 96))
    image_np = np.array(image, dtype=np.float32)
    image_np = (image_np / 255.0 - 0.0979) / 0.1991
    input_exponent = -13
    image_np = np.clip(image_np * (1 << -input_exponent), -32768, 32767).astype(np.int16)

    print("{", end="")
    for i, pixel in enumerate(image_np.flatten()):
        print(f"{pixel}", end=", " if i < len(image_np.flatten()) - 1 else "")
    print("}")

preprocess_image(r'D:\Projects\HandRecog_cnn\dataset\test\08\07_ok\frame_08_07_0159.png')

  - 定义输入张量
Tensor<int16_t> input;
input.set_element((int16_t *)example_element)
     .set_exponent(input_exponent)
     .set_shape({input_height, input_width, input_channel})
     .set_auto_free(false);
- 推理
  - 初始化模型。
HANDRECOGNITION model;
  - 使用 Latency 对象测量模型推理的延迟（前向传播时间）。
  - 调用 model.forward(input) 进行推理
dl::tool::Latency latency;
latency.start();
model.forward(input);
latency.end();
latency.print("\nHANDRECOGNITION", "forward");
  - 预测模型结果
    - 获取输出
float *score = model.l14.get_output().get_element_ptr();
    - 初始化max_score变量为模型输出的第一个元素值，作为初始比较值
    - 循环，比较索引元素值，更新max_score
    在循环结束后，max_index将指向模型认为最有可能的类别索引，而max_score则是这个类别的置信度得分
float max_score = score[0];
int max_index = 0;

// 计算每种手势的得分
for (size_t i = 0; i < 10; i++) 
{
    printf("%f, ", score[i] * 100);
    if (score[i] > max_score)
    {
        max_score = score[i];
        max_index = i;
    }
}
printf("\n");

// 输出预测结果
switch (max_index)
{
    case 0:
        printf("01_palm\n");
        break;
    case 1:
        printf("02_I\n");
        break;
    case 2:
        printf("03_fist\n");
        break;
    case 3:
        printf("04_fist_moved\n");
        break;
    case 4:
        printf("05_thumb\n");
        break;
    case 5:
        printf("06_index\n");
        break;
    case 6:
        printf("07_ok\n");
        break;
    case 7:
        printf("08_palm_moved\n");
        break;
    case 8:
        printf("09_c\n");
        break;
    case 9:
        printf("10_down\n");
        break;
    default:
        printf("No result\n");
}
6.3 部署模型
[] idf.py set-target esp32s3
运行后，终端没有任何输出
使用 python  idf.py的绝对路径      set-target esp32s3
[] idf.py build
1. 相关库文件无法正确导入
[图片]
[图片]
在c_cpp_properties.json增加includePath
[图片]

---
2. 问题仍未解决
增加的路径使用完整路径

---
3. 分区表配置问题
[图片]
分区表的配置是编译后生成的
Generating ../../partition_table/partition-table.bin
Partition table binary generated. Contents:
ESP-IDF Partition TablName, Type, SubType, Offset, Size, Flags
nvs,data,nvs,0x9000,24K,
phy_init,data,phy,0xf000,4K,
factory,app,factory,0x10000,1M,
解决方法：
- 创建一个partitions.csv
# Name,   Type, SubType, Offset,  Size, Flags
factory, app,  factory, 0x010000, 3840K
nvs,     data, nvs,     0x3D0000, 16K
fr,      data,   ,      0x3E0000, 128K
ota_0,   app,  ota_0,   0x400000, 3072K  
ota_1,   app,  ota_1,   0x700000, 3072K 
- idf.py menuconfig
  1. 更改flash size
  2. 使用自己创建的partitions.csv
  3. Ram config
[图片]
[图片]
[图片]

---
4. 分区表除了相关配置，不要增加注释
否则会出现：
[图片]
5. 看门狗设置时间超时
[图片]
也是idf.py menuconfig
Component config →esp system setting → watchdog相关，设置10s/20s

---
成功后出现：
[图片]
[] idf.py -p  port  flash
[] idf.py -p  port  monitor
[图片]
esp32相关实践
搭建网络服务器
1. 
#include <WiFi.h>
#include <WebServer.h>

const char*ssid="Hatty的P60";
const char*password="mssqqrdtdifhvgn";

WebServer server(80);

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid,password);

  int i=0;
  while(WiFi.status()!=WL_CONNECTED)){
    delay(1000);
    Serial.print(i++);
    Serial.print(' ');
  }
  Serial.println("Connected to WiFi.");
  Serial.println(WiFi.localIP());

  server.on("/",handleRoot);
  server.onNotFound(handleNotFound);

  server.begin();

}

void handleRoot(){
  server.send(200,"text/plain","Hello World");
}

void handleNotFound(){
  server.send(404,"text/plain","404:Not Found");
}

void loop() {
  server.handelClient();

}

[图片]
[图片]

---
2. 拍摄照片
这部分还未应用，待更新。。。
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// 替换为你的 WiFi SSID 和密码
const char* ssid = "your_SSID";
const char* password = "your_PASSWORD";

WebServer server(80);

void handleRoot() {
  server.send(200, "text/plain", "Hello World");
}

void handleNotFound() {
  server.send(404, "text/plain", "404: Not Found");
}

void handleCapture() {
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.onNotFound(handleNotFound);

  server.begin();

  // 初始化摄像头
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // 摄像头初始化
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
}

void loop() {
  server.handleClient();
}

esp-idf扩展的基本用法
安装和配置
1. 安装 ESP-IDF 扩展：
  - 打开 VSCode，进入扩展商店，搜索并安装 Espressif IDF 扩展。
2. 安装 ESP-IDF 工具链：
  - 按 Ctrl + Shift + P 打开命令面板，输入 ESP-IDF: Configure ESP-IDF extension 并按提示进行安装和配置。
  - 安装工具链、Python、Git 和其他依赖项。
3. 配置 ESP-IDF：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Configure Paths 设置相关路径，如 ESP-IDF 路径、工具路径等。
创建和配置项目
1. 创建新项目：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Create ESP-IDF project。
  - 选择模板、项目名称和路径。
2. 配置项目：
  - 打开命令面板，输入 ESP-IDF: SDK Configuration Editor（或 Ctrl + E G）打开 menuconfig。
  - 配置目标板、串口设置、Flash 设置等。
  - 按 S 保存，按 Q 退出。
编译和烧录项目
1. 编译项目：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Build your project（或 Ctrl + E B）进行编译。
2. 选择 Flash 方法：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Select Flash Method，选择 UART 或 JTAG。
3. 烧录项目：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Flash your project（或 Ctrl + E F）进行烧录。
调试和监控
1. 启动调试：
  - 按 F5 启动调试会话。配置 launch.json 以设置调试选项。
2. 监视串口输出：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Monitor your project（或 Ctrl + E M）查看串口输出。
  - 在终端中按 Ctrl + C 停止监视器。
其他实用命令
1. 清理项目：
  - 打开终端，导航到项目目录，运行 idf.py clean 清理生成文件。
2. 打开终端：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Open ESP-IDF Terminal 打开带有 ESP-IDF 环境的终端。
3. 检查和修复项目配置：
  - 按 Ctrl + Shift + P，输入 ESP-IDF: Doctor Command 检查环境配置，自动修复常见问题。

---
遇到问题
1. 一开始希望esp-idf项目使用的python环境也是模型训练的环境（使用anaconda虚拟环境），但是新建esp-idf项目的过程会自动生成一个python（较新版本）虚拟环境
更改python路径
[图片]
2. anaconda的python环境中不含有相关工具链
[图片]
于是我重新运行esp-idf中的install.bat
生成了和anaconda虚拟环境一样版本的python虚拟环境，并且安装了相关工具链，于是我把路径更改为该python虚拟环境的路径