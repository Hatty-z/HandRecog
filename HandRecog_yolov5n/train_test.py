import yaml
import torch
from yolov5.train import train  # 从你上传的 train.py 文件中导入 train 函数
from yolov5.val import run as val_run  # 从你上传的 val.py 文件中导入 run 函数

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    data_config_path = r'D:\Projects\HandRecog\dataset\data.yaml'
    weights = 'yolov5n.pt'  
    epochs = 30
    batch_size = 4
    img_size = 320

    # 检查是否可以使用 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置训练参数
    train_params = {
        'data': data_config_path,
        'weights': weights,
        'epochs': epochs,
        'batch_size': batch_size,
        'imgsz': img_size,
        'device': device,
        'name': 'yolov5n_hand_gesture',
        'patience': 5,
        'hyp': None,
        'save_period': -1,
        'single_cls': False,
        'optimizer': 'auto',
        'sync_bn': False,
        'workers': 8,
        'project': 'runs/train',
        'exist_ok': False,
        'resume': False,
        'nosave': False,
        'noval': False,
        'noautoanchor': False,
        'evolve': False,
        'bucket': '',
        'cache_images': False,
        'image_weights': False,
        'rect': False,
        'multi_scale': False,
        'adam': False,
        'sync_bn': False,
        'local_rank': -1,
        'entity': None,
        'upload_dataset': False,
        'bbox_interval': -1,
        'artifact_alias': 'latest'
    }

    # 调用训练函数
    train(**train_params)

    # 进行验证
    val_metrics = val_run(
        data=data_config_path,
        weights=weights,
        batch_size=batch_size,
        imgsz=img_size,
        device=device,
        project='runs/val',
        name='yolov5n_hand_gesture_val',
        exist_ok=True
    )
    print(f"Validation results: {val_metrics}")

if __name__ == '__main__':
    main()
