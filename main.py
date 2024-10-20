import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mobilenet_sigmoid.mobilenet import MobileNet
from mobilenet_sigmoid.data import get_train_valid_loader, get_test_loader


def train(model, train_loader, criterion, optimizer, device, save_l2_norm=False):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    # l2_norms = []
    l2_norm = None
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 计算并保存 L2 范数
        if save_l2_norm:
            l2_norm = calculate_l2_norm(model.layers[8].conv1.weight.grad)
            # l2_norms.append(l2_norm)
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    if save_l2_norm:
        return epoch_loss, epoch_acc, l2_norm
    else:
        return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)

    return epoch_loss, epoch_acc


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            running_corrects += (predicted == target).sum().item()

    test_loss /= total
    accuracy = running_corrects / total
    return test_loss, accuracy


def save_results(results, base_path, experiment_name):
    # 创建目录
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_path = os.path.join(base_path, f"{experiment_name}.txt")
    with open(file_path, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

def calculate_l2_norm(tensor):
    return torch.norm(tensor, p=2).item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, valid_loader = get_train_valid_loader(
        args.data_dir, args.batch_size, args.augment, args.random_seed
    )
    test_loader = get_test_loader(args.data_dir, args.batch_size)

    # 模型初始化
    model = MobileNet(num_classes=100, sigmoid_block_ind=args.sigmoid_block_ind).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    best_acc = 0.0
    results = []
    l2_norms = []
    for epoch in range(args.epochs):
        if args.save_l2_norm:
            train_loss, train_acc, l2_norm = train(model, train_loader, criterion, optimizer, device, save_l2_norm=True)
            l2_norms.append(l2_norm)
        else:
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch}/{args.epochs - 1}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')

        # 保存结果
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'valid_loss': valid_loss,
            'valid_acc': valid_acc.item()
        })

        if valid_acc > best_acc:
            best_acc = valid_acc

    if args.save_model:
        # 创建保存模型的目录
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)

        # 构建完整的文件路径
        model_path = os.path.join(model_dir, f'{args.name_model}.pth')
        # 保存模型
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    results.append({
        'Best Validation Accuracy': best_acc.item()
    })

    if args.test:
        test_loss, test_acc = test(model, test_loader, criterion, device)
        results.append({
            'Test Accuracy': test_loss,
            'Test Loss': test_acc
        })

    # 将结果写入文件
    if args.save_l2_norm:
        np.save(os.path.join(args.output_dir, 'l2_norms.npy'), np.array(l2_norms))
    save_results(results, args.output_dir, args.output_file)

    print(f'Best Validation Accuracy: {best_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MobileNet Training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--use-scheduler', action='store_true', help='Use cosine annealing scheduler')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--sigmoid-block-ind', type=int, nargs='+', default=[],
                        help='Indices of blocks to use Sigmoid activation')
    parser.add_argument('--save-l2-norm', action='store_true', help='Save L2 norm of gradients')
    parser.add_argument('--save-model', action='store_true', help='Save Models')
    parser.add_argument('--name-model', type=str, default='best_model',
                        help='Name of saved models')
    parser.add_argument('--test', action='store_true', help='Test with test dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--output-file', type=str, required=True, help='File name to save results')

    args = parser.parse_args()
    main(args)
