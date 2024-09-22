# Train Hand_Gesture_Recognition Model and Deployed on ESP32-S3

## Abstract:
1. Build and train a CNN model for hand gesture recognition (attempts were made with YOLOv8n and YOLOv5n models).
2. Optimize, quantize, and convert the trained model to a suitable format for deployment on the ESP32-S3.
3. Connect the ESP32-S3 CAM with an OV2640 module, set up a web server to capture images, and recognize the gestures in the captured images.

[kaggle下载手势识别数据集(dataset)](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
learn more in [my blog](https://hatty.top/2024/09/17/%E6%89%8B%E5%8A%BF%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E5%9C%A8esp32s3%EF%BC%88%E4%B8%80%EF%BC%89/)

## 手势识别模型
### 1. 数据
1. 标注
2. 划分数据集
- 这里划分了训练集和测试集，校准集从训练集和测试集中随机抽取
- 划分校准集并存储为.pkl文件
3. 数据预处理（包含在模型训练脚本中）

### 2. 模型选择与设计
1. cnn模型
 - HandGestureCNN 类
 - EarlyStopping 类 
2. yolov8n
yolov8n和yolov5n进行了尝试，但是在部署阶段，由于模型较为复杂，很多网络层espdl不支持
可能需要重新定义模型
但是模型部署前的训练都已完成
- 数据标注
YOLO的标签文件需要每一行包含如下信息：
<class_id> <x_center> <y_center> <width> <height>

3. yolov5n
```
cd D:/Projects/HandRecog/yolov5
python train.py --img 320 --epochs 1 --data ../dataset/data.yaml --weights yolov5n.pt --batch-size 4 --patience 5 --hyp D:/Projects/HandRecog/yolov5/data/hyps/hyp.scratch-low.yaml --project D:/Projects/HandRecog --name runs
```
### 3. 模型训练,验证与调优
1. 准备数据集
2. 训练
实例化，加载模型，初始化参数（训练轮数，学习率，早停机制），开始训练
3. 评估
对于测试集的图片
得到模型输出，预测标签
累计总数，正确预测的数量
计算损失并累加
计算准确率，平均损失
4. 调优
更新学习率调度器
检查是否触发早停机制
保存模型状态
5. 结果
[图片]
### 4. 模型转换
1. 转换为onnx格式模型
2. 转换为esp-dl适配模型（使用量化工具包）
   - 模型优化
   - 模型量化
   - 校准

### 5. 模型部署1
esp-idf安装
>模型部署1与2思路做法基本一致，只有模型定义不太一样
>- 模型部署2流程完整，但是最终的识别准确率不高
>- 模型部署1出现，最终监视输出时出现5.3部署模型中的问题（尚未解决，其他你的解决方案），也可以把部署流程走一遍，但是看不到结果www

1. 构建模型
- 实现build函数
- 实现call函数
2. 运行模型
- 创建模型对象
- 定义输入
- 推理
3. 部署模型
idf.py set-target esp32s3
idf.py build
idf.py -p  port  flash
idf.py -p  port  monitor
### 6. 模型部署2


## esp32相关实践
1. 搭建网络服务器
2. 拍摄照片
这部分还未应用，待更新。。。
esp-idf扩展的基本用法