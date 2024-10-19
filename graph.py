import glob

import matplotlib.pyplot as plt
import os

import numpy as np

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25


box_colors = ['#2878B5', '#9AC9DB', '#F8AC8C', '#C82423', '#FF8884']
# box_colors = ['#A2BDDB', '#8F89AE', '#65B465', '#EFD07D', '#FF6666']

def read_results_from_file(file_path):
    """读取文件内容并解析为结果列表"""
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            # 将每行转换为字典并添加到结果列表中
            results.append(eval(line.strip()))
    return results


def plot_results_from_files(file_paths, label_interval=5, show=True):
    """
    从文件路径列表中读取结果并绘制图表。

    参数：
    - file_paths: 文件路径列表。
    - label_interval: 每隔多少个 epoch 标注一次损失和准确率。
    """

    for file_path in file_paths:
        # 读取文件名作为图表标题
        # title = os.path.basename(file_path)[:-4] # .split('.')[0]
        title = 'activation=ReLU'
        # 读取结果
        try:
            results = read_results_from_file(file_path)[:-1]

            # 提取数据
            epochs = [entry['epoch'] for entry in results]
            train_losses = [entry['train_loss'] for entry in results]
            valid_losses = [entry['valid_loss'] for entry in results]
            train_accs = [entry['train_acc'] for entry in results]
            valid_accs = [entry['valid_acc'] for entry in results]
        except:
            results = read_results_from_file(file_path)[:-2]

            # 提取数据
            epochs = [entry['epoch'] for entry in results]
            train_losses = [entry['train_loss'] for entry in results]
            valid_losses = [entry['valid_loss'] for entry in results]
            train_accs = [entry['train_acc'] for entry in results]
            valid_accs = [entry['valid_acc'] for entry in results]


        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # 绘制损失图
        ax1.plot(epochs, train_losses, label='Training Loss', color=box_colors[0])
        ax1.plot(epochs, valid_losses, label='Validation Loss', color=box_colors[2])
        label_interval = int(len(epochs) / 10)
        for i in range(0, len(epochs), label_interval):
            ax1.text(epochs[i], train_losses[i], f'{train_losses[i]:.2f}', color=box_colors[0], fontsize=15, ha='right')
            ax1.text(epochs[i], valid_losses[i], f'{valid_losses[i]:.2f}', color=box_colors[2], fontsize=15, ha='right')
        if len(epochs) // label_interval != 0:
            epoch = len(epochs) -1
            ax1.text(epochs[epoch], train_losses[epoch], f'{train_losses[epoch]:.2f}', color=box_colors[0], fontsize=15,
                     ha='right')
            ax1.text(epochs[epoch], valid_losses[epoch], f'{valid_losses[epoch]:.2f}', color=box_colors[2], fontsize=15,
                     ha='right')
        ax1.set_title(f'Loss over Epochs - {title}', fontsize=23)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        # ax1.grid(True)

        # 绘制准确率图
        ax2.plot(epochs, train_accs, label='Training Accuracy', color=box_colors[1])
        ax2.plot(epochs, valid_accs, label='Validation Accuracy', color=box_colors[3])
        for i in range(0, len(epochs), label_interval):
            ax2.text(epochs[i], train_accs[i], f'{train_accs[i]:.2f}', color=box_colors[1], fontsize=15, ha='right')
            ax2.text(epochs[i], valid_accs[i], f'{valid_accs[i]:.2f}', color=box_colors[3], fontsize=15, ha='right')
        if len(epochs) // label_interval != 0:
            epoch = len(epochs) - 1
            ax2.text(epochs[epoch], train_accs[epoch], f'{train_accs[epoch]:.3f}', color=box_colors[1], fontsize=15,
                     ha='right')
            ax2.text(epochs[epoch], valid_accs[epoch], f'{valid_accs[epoch]:.3f}', color=box_colors[3], fontsize=15,
                     ha='right')
        ax2.set_title(f'Accuracy over Epochs - {title}', fontsize=23)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        # ax2.grid(True)

        # 调整布局
        plt.tight_layout()
        plt.savefig(f'graphs/{title}.pdf')
        if show:
            plt.show()
        plt.clf()


def plot_l2_norms(l2_norms_file, output_image_file):
    # 读取 L2 范数
    l2_norms = np.load(l2_norms_file)

    # 绘制图表
    plt.figure(figsize=(10, 7))
    plt.plot(l2_norms, label='L2 Norm of Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.title('L2 Norm of Gradients for model.layers[8].conv1.weight')
    plt.legend()
    # plt.grid(True)
    plt.savefig(output_image_file)
    plt.show()


def get_all_file_paths(directory):
    all_files = glob.glob(os.path.join(directory, '**', '*.*'), recursive=True)
    base_dir = os.path.dirname(directory)  # 获取父目录
    relative_paths = [os.path.relpath(file, start=base_dir) for file in all_files]
    normalized_paths = [path.replace(os.sep, '/') for path in relative_paths]
    normalized_paths.sort()
    return normalized_paths

if __name__ == '__main__':
    # file_path = 'experiment/activation_function/l2_norms.npy'
    # plot_l2_norms(file_path, 'graphs/l2_norm.png')
    # 文件路径列表
    file_paths = [
        # 'experiment/learning_rate/lr=0.01.txt',
        # 'experiment/learning_rate/lr=0.2.txt',
        # 'experiment/learning_rate/lr=0.05.txt',
        # 'experiment/activation_function/activation=sigmoid.txt',
        # 'experiment/weight_decay/weight_decay=1e-4.txt',
        'experiment/weight_decay/weight_decay=5e-4.txt'
        # 'experiment/learning_rate_schedule/scheduler=cosine.txt',
        # 'experiment/learning_rate_schedule/scheduler=constant.txt',
    ]
    # directory = 'experiment/weight_decay'
    # file_paths = get_all_file_paths(directory)
    # 调用函数绘制图表
    plot_results_from_files(file_paths, label_interval=3)
