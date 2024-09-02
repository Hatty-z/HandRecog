import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from model import HandGestureCNN, EarlyStopping  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_dir = 'runs/exp'
writer = SummaryWriter(log_dir)

# 自定义数据集类
class GestureDataset(Dataset):
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
    
    def __len__(self):
        return len(self.img_labels)

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

#准备数据集
train_dataset = GestureDataset('/root/autodl-tmp/HandRecog_cnn/dataset/train', transform=transform)
test_dataset = GestureDataset('/root/autodl-tmp/HandRecog_cnn/dataset/test', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
