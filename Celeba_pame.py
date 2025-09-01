# train_celeba_adam.py
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import time
import pandas as pd
from PIL import Image
import numpy as np
from pame import PAME


# 自定义CelebA数据集类（支持CSV格式）
class CelebADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.transform = transform

        # 检查图像目录是否存在
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图像目录未找到: {self.img_dir}")

        # 尝试找到属性文件（支持多种格式）
        attr_files = [
            'list_attr_celeba.csv',  # CSV格式
            'list_attr_celeba.txt',  # TXT格式
            'celeba_attributes.csv',  # 其他可能的名称
            'attributes.csv'
        ]

        self.attr_path = None
        for file in attr_files:
            path = os.path.join(root_dir, file)
            if os.path.exists(path):
                self.attr_path = path
                print(f"使用属性文件: {file}")
                break

        if self.attr_path is None:
            raise FileNotFoundError(f"未找到属性文件! 在 {root_dir} 目录下检查了: {attr_files}")

        # 根据文件扩展名选择合适的读取方式
        if self.attr_path.endswith('.csv'):
            # 读取CSV格式的属性文件
            df = pd.read_csv(self.attr_path)

            # 检查列名格式并处理
            if 'image_id' in df.columns:
                df.set_index('image_id', inplace=True)
            elif 'filename' in df.columns:
                df.set_index('filename', inplace=True)
            elif df.columns[0].lower() == 'unnamed: 0':
                df.set_index(df.columns[0], inplace=True)
        else:
            # 读取TXT格式的属性文件
            df = pd.read_csv(self.attr_path, sep=r'\s+', header=1, index_col=0)

        # 确保索引是字符串类型
        df.index = df.index.astype(str)
        df.replace(to_replace=-1, value=0, inplace=True)

        # 划分数据集
        n_total = len(df)
        if split == 'train':
            self.df = df.iloc[:162770]  # 前80%
        elif split == 'valid':
            self.df = df.iloc[162770:182637]  # 中间10%
        else:  # 'test'
            self.df = df.iloc[182637:]  # 后10%

        self.attr_names = list(df.columns)
        self.filenames = [str(idx) for idx in self.df.index]
        print(f"已加载 {split} 数据集: {len(self.df)} 样本")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]

        # 确保文件名有.jpg扩展名
        if not img_name.lower().endswith('.jpg'):
            img_name += '.jpg'

        img_path = os.path.join(self.img_dir, img_name)

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            # 尝试其他可能的扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = os.path.join(self.img_dir, img_name.split('.')[0] + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"图像文件未找到: {img_path}")

        image = Image.open(img_path).convert('RGB')

        # 获取属性 (40个二值属性)
        attrs = self.df.iloc[idx].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, attrs


# 创建ResNet18模型（适配CelebA）
def create_model(num_classes=40):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():

    # 设置随机种子
    torch.manual_seed(42)

    # 设置本地数据集路径
    data_root_options = [
        './data/celeba',  # 默认路径
        './data',  # 如果celeba文件夹不存在但文件在data目录下
        'D:/PythonProject/Face/data/celeba'  # 示例绝对路径
    ]

    # 尝试找到存在的路径
    data_root = None
    for path in data_root_options:
        if os.path.exists(path):
            # 检查该路径下是否有图像文件夹
            if os.path.exists(os.path.join(path, 'img_align_celeba')):
                data_root = path
                print(f"使用数据集路径: {data_root}")
                break

    if data_root is None:
        raise FileNotFoundError("未找到CelebA数据集! 请检查路径或下载数据集")

    # 数据预处理
    print("准备数据加载器...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集
    try:
        train_dataset = CelebADataset(data_root, split='train', transform=transform)
        test_dataset = CelebADataset(data_root, split='test', transform=transform)
    except Exception as e:
        print(f"数据集创建失败: {e}")
        print("请确保数据集包含以下内容：")
        print("- img_align_celeba/ 文件夹（包含所有图像）")
        print("- 属性文件 (CSV或TXT格式)")
        return

    # 数据加载器
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)


    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: ResNet18 (适配CelebA)")
    print(f"总参数量: {total_params:,}")
    print(f"使用设备: {device}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")

    optimizer = PAME(
        model.parameters(),
        lr=0.001,
        base_beta1=0.9,
        base_beta2=0.999,
        gamma=0.995
    )

    criterion = nn.BCEWithLogitsLoss()

    # 训练函数
    def train(epoch):
        model.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()

        print(f"\nEpoch {epoch + 1} 训练中...")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # 记录批次级指标到WandB
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / total_samples


            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = total_loss / total_samples
                elapsed = time.time() - start_time
                print(f"批次 [{batch_idx + 1}/{len(train_loader)}] | "
                      f"耗时: {elapsed:.2f}s | "
                      f"损失: {avg_loss:.4f}")

        return total_loss / total_samples

    # 测试函数
    def test():
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)

                # 计算准确率
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == targets.bool()).sum().item()
                total += targets.numel()

        accuracy = 100. * correct / total
        return total_loss / len(test_loader.dataset), accuracy

    # 主训练循环
    best_acc = 0
    epochs = 10
    start_time = time.time()

    print("\n开始训练CelebA...")
    for epoch in range(epochs):
        train_loss = train(epoch)
        test_loss, test_acc = test()

        # 获取当前β值
        beta1, beta2 = optimizer.get_current_betas()


        print(f"\nEpoch {epoch + 1}/{epochs}: "
            f"训练损失={train_loss:.4f} | "
            f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.2f}%")
        print(f"当前β值: beta1={beta1:.4f}, beta2={beta2:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'best_model_celeba_pame.pth')
            print(f"保存新最佳模型，准确率: {test_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
    print(f"总训练时间: {training_time / 60:.2f} 分钟")
    print(f"平均每epoch时间: {training_time / epochs:.2f} 秒")


    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'final_model_celeba_pame.pth')



if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()