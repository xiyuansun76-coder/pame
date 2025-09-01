# train_cifar10.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pame import PAME
from torch.utils.data import DataLoader
import multiprocessing
import os
import time
import numpy as np

# 定义SimpleCNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # 1. 准备数据集
    print("准备数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 根据CPU核心数设置工作进程数
    num_workers = min(4, os.cpu_count())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 2. 创建模型并转移到CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: SimpleCNN")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"使用设备: {device}")
    print(f"数据加载器工作进程数: {num_workers}")

    # 3. 初始化PAME优化器
    optimizer = PAME(model.parameters(),
                     lr=0.001,
                     base_beta1=0.9,
                     base_beta2=0.999,
                     gamma=0.995)

    criterion = nn.CrossEntropyLoss()

    # 4. 训练函数
    def train(epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        batch_count = len(train_loader)

        print(f"\nEpoch {epoch + 1} 训练中...")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # 统计信息
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == batch_count:
                avg_loss = total_loss / total
                accuracy = 100. * correct / total
                elapsed = time.time() - start_time
                print(f"批次 [{batch_idx + 1}/{batch_count}] | "
                      f"耗时: {elapsed:.2f}s | "
                      f"损失: {avg_loss:.4f} | "
                      f"准确率: {accuracy:.2f}%")

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    # 5. 测试函数
    def test():
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    # 6. 主训练循环
    best_acc = 0
    epochs = 10
    start_time = time.time()

    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train(epoch)
        # 测试
        test_loss, test_acc = test()

        # 获取当前β值
        beta1, beta2 = optimizer.get_current_betas()

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}% | "
              f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.2f}%")
        print(f"当前β值: beta1={beta1:.4f}, beta2={beta2:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'best_model_cifar10.pth')
            print(f"保存新最佳模型，准确率: {test_acc:.2f}%\n")
        else:
            print()

    training_time = time.time() - start_time
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
    print(f"总训练时间: {training_time / 60:.2f} 分钟")
    print(f"平均每epoch时间: {training_time / epochs:.2f} 秒")

    # 7. 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'final_model_cifar10.pth')

if __name__ == '__main__':
    # Windows平台需要添加freeze_support
    multiprocessing.freeze_support()

    # 设置多进程启动方法
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()