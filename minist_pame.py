# train_mnist_pame.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import csv
from pame_optimizer import PAME


# 简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    # 设置随机种子
    torch.manual_seed(42)

    # MNIST数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])

    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    # 根据CPU核心数设置工作进程数
    num_workers = min(4, os.cpu_count())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 模型和设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: 简单CNN")
    print(f"总参数量: {total_params:,}")
    print(f"使用设备: {device}")
    print(f"数据加载器工作进程数: {num_workers}")

    # 使用PAME优化器
    optimizer = PAME(model.parameters(),
                     lr=0.001,
                     base_beta1=0.9,
                     base_beta2=0.999,
                     gamma=0.995)

    criterion = nn.CrossEntropyLoss()

    # 训练函数
    def train(epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 使用普通循环替代tqdm
        print(f"\nEpoch {epoch + 1} 训练中...")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / total
                accuracy = 100. * correct / total
                print(f'批次 [{batch_idx}/{len(train_loader)}], '
                      f'损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')

        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    # 测试函数
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

        return total_loss / total, 100. * correct / total

    # 创建CSV文件记录结果
    with open('mnist_pame_results.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])

        # 主训练循环
        best_acc = 0
        epochs = 10
        start_time = time.time()

        print("\n开始训练MNIST...")
        for epoch in range(epochs):
            train_loss, train_acc = train(epoch)
            test_loss, test_acc = test()

            # 获取当前β值
            beta1, beta2 = optimizer.get_current_betas()

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}% | "
                  f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.2f}%")
            print(f"当前β值: beta1={beta1:.4f}, beta2={beta2:.4f}")

            # 写入CSV
            csv_writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                test_loss,
                test_acc
            ])

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                }, 'mnist_pame_best.pth')
                print(f"保存新最佳模型，准确率: {test_acc:.2f}%\n")
            else:
                print()

        training_time = time.time() - start_time
        print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
        print(f"总训练时间: {training_time / 60:.2f} 分钟")

    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'mnist_pame_final.pth')


if __name__ == '__main__':
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()