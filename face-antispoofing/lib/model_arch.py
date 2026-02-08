import torch.nn as nn
import torch
import torchvision.models as tm
import torch.nn.functional as F
import numpy as np
import random


class ROI_Pooling(nn.Module):
    '''
    处理单个feature map的 roi 图像信息
    '''

    def __init__(self):
        super().__init__()
        self.avgpool_patch = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_patch = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, feature_map, cluster_center, spatial_ratio):
        feature_list = []
        cluster_center_mean = torch.mean(cluster_center, dim=0)
        cluster_center_normal = cluster_center_mean / spatial_ratio
        cluster_center_int = torch.floor(cluster_center_normal)
        cluster_center_float = cluster_center_normal - cluster_center_int
        cluster_center_offset = torch.round(cluster_center_float)
        cluster_center_offset = cluster_center_offset * 2 - 1  # 转到[-1,1]
        cluster_center_int = cluster_center_int + 1  # 转到[1,5]
        cluster_center_int = cluster_center_int + cluster_center_offset

        padding = (1, 1, 1, 1)
        # feature_map = F.pad(feature_map, padding, 'constant', 1)

        # for index in range(cluster_center_mean.shape[0]):
        #     coordinate_single = cluster_center_int[index]
        #     coordinate_single=coordinate_single.long()
        #     # x2 是因为python 索引的问题,从0开始,[0:1] 只索引一个
        #
        #     patch = feature_map[:, :,
        #                         coordinate_single[0]:coordinate_single[0] + 2,
        #                         coordinate_single[1]:coordinate_single[1] + 2]
        #
        patch_avg = self.avgpool_patch(feature_map)
        patch_max = self.maxpool_patch(feature_map)
        patch_feature = patch_avg
        patch_flatten = torch.flatten(patch_feature, 1)
        feature_list.append(patch_flatten)

        return feature_list


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        padding = 0

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.avg = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def modality_drop_v1(x_rgb, x_ir, x_depth, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]

    if p == [0, 0, 0]:
        # print("drop")
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")

        prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        prob = np.array((0, 0, 0, 0, 0, 0, 1))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_ir = x_ir * p[:, 1]
    x_depth = x_depth * p[:, 2]
    p = p.squeeze()
    return x_rgb, x_ir, x_depth, p


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import time
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
import warnings

def universal_tsne_visualizer(input_features, labels=None, save_path='tsne_plot.png',
                              perplexity=30, learning_rate=200, n_iter=1000,
                              point_size=8, alpha=0.7, random_state=42,
                              apply_scaling=True, max_samples=5000):
    """
    通用t-SNE可视化工具 - 支持任意形状特征输入

    参数:
    input_features -- 输入特征（支持多种格式）:
        - NumPy数组（1D, 2D, 3D）
        - PyTorch/TensorFlow张量
        - 稀疏矩阵
        - 字典/列表/DataFrame
        - 任何可迭代对象
    labels -- 可选标签（默认：None）
    save_path -- 图片保存路径（默认：'tsne_plot.png'）
    perplexity -- t-SNE困惑度（默认：30）
    learning_rate -- 学习率（默认：200）
    n_iter -- 迭代次数（默认：1000）
    point_size -- 点大小（默认：8）
    alpha -- 透明度（默认：0.7）
    random_state -- 随机种子（默认：42）
    apply_scaling -- 是否应用特征缩放（默认：True）
    max_samples -- 最大样本数（超过则采样）（默认：5000）

    返回：降维后的二维坐标 (n_samples, 2)
    """
    start_time = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*'n_iter'.*")

    # 1. 转换为适合t-SNE的格式
    #print("\n🔧 预处理特征数据...")
    features, processed_labels = preprocess_input(input_features, labels, max_samples)

    # 2. 特征预处理
    #print(f"📊 输入形状: {features.shape} | 数据类型: {features.dtype}")

    # 应用标准化
    if apply_scaling:
        #print("🔢 应用标准化（均值为0，方差为1）")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # 3. 处理大数据集
    if features.shape[0] > max_samples:
        print(f"📈 数据集较大（{features.shape[0]}样本），采样至{max_samples}个样本")
        indices = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[indices]
        if processed_labels is not None:
            processed_labels = processed_labels[indices]

    # 4. 执行t-SNE
    # print("\n🌀 执行t-SNE降维...")
    # print(f"样本数: {features.shape[0]} | 特征数: {features.shape[1]}")
    # print(f"超参数: 困惑度={perplexity}, 学习率={learning_rate}, 迭代次数={n_iter}")

    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=n_iter,
                random_state=random_state,
                n_jobs=-1)  # 使用所有CPU核心

    embedding = tsne.fit_transform(features)

    elapsed = time.time() - start_time
    #print(f"✅ t-SNE完成! 耗时: {elapsed:.2f}秒")

    # 5. 可视化
    #print("\n🎨 创建可视化...")
    plt.figure(figsize=(10, 8))

    # 有标签时使用分类着色
    if processed_labels is not None:
        plot_with_labels(embedding, processed_labels, point_size, alpha)
    # 无标签时使用单一颜色
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    s=point_size, alpha=alpha,
                    color='royalblue')

    # 添加标题和标签
    # 修改标题和坐标轴标签为英文
    plt.title(f't-SNE Visualization (n={embedding.shape[0]}, dim={features.shape[1]})', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.2)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"\n💾 t-SNE图已保存至: {os.path.abspath(save_path)}")
    plt.close()

    return embedding


def preprocess_input(features, labels, max_samples):
    """将各种格式的输入转换为标准的2D特征矩阵"""
    # 处理标签
    processed_labels = None
    if labels is not None:
        processed_labels = convert_to_numpy(labels).flatten()

    # 处理各种特征类型
    # 1. 已经是2D numpy数组
    if isinstance(features, np.ndarray) and features.ndim == 2:
        return features, processed_labels

    # 2. 将其他格式转换为numpy
    processed_features = convert_to_numpy(features)

    # 3. 处理1D数组（单个样本）
    if processed_features.ndim == 1:
        processed_features = processed_features.reshape(1, -1)

    # 4. 处理3D或更高维数组（如BCHW图像特征）
    if processed_features.ndim > 2:
        #print(f"⚠️ 检测到{processed_features.ndim}维输入，展平为2D矩阵")
        original_shape = processed_features.shape
        processed_features = processed_features.reshape(original_shape[0], -1)
        #print(f"  原始形状: {original_shape} -> 新形状: {processed_features.shape}")

    # 5. 处理稀疏矩阵
    if issparse(processed_features):
        print(f"⚠️ 检测到稀疏矩阵（{type(processed_features)}），转换为密集格式")
        processed_features = processed_features.toarray()

    # 6. 确保至少2维
    if processed_features.ndim != 2:
        raise ValueError(f"无法将输入转换为2D矩阵。最终维度: {processed_features.ndim}")

    return processed_features, processed_labels


def convert_to_numpy(data):
    """将各种数据类型转换为NumPy数组"""
    # 1. 已经是numpy数组
    if isinstance(data, np.ndarray):
        return data

    # 2. PyTorch张量
    if hasattr(data, 'detach') and hasattr(data, 'numpy'):
        return data.detach().cpu().numpy()

    # 3. TensorFlow张量
    if hasattr(data, 'numpy'):
        return data.numpy()

    # 4. pandas DataFrame/Series
    if hasattr(data, 'values'):
        return data.values

    # 5. 稀疏矩阵
    if issparse(data):
        return data

    # 6. 字典类型（键作为特征）
    if isinstance(data, dict):
        return np.array(list(data.values()))

    # 7. 列表或元组
    if isinstance(data, (list, tuple)):
        return np.array(data)

    # 8. 单个值（标量）
    try:
        scalar = float(data)
        return np.array([scalar])
    except:
        pass

    # 9. 其他可迭代对象
    try:
        return np.array([x for x in data])
    except:
        pass

    raise TypeError(f"不支持的输入类型: {type(data)}")


def plot_with_labels(embedding, labels, point_size, alpha):
    """带标签的t-SNE可视化"""
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 自动选择绘图策略
    if num_classes <= 12:
        # 类别少时使用不同颜色
        plot_colored(embedding, labels, point_size, alpha)
    elif num_classes <= 30:
        # 中等类别使用颜色+形状
        plot_colored_with_shapes(embedding, labels, point_size, alpha)
    else:
        # 大量类别使用连续色谱
        plot_continuous(embedding, labels, point_size, alpha)


def plot_colored(embedding, labels, point_size, alpha):
    """类别少于12个时的着色方案"""
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        # 为不同类别使用不同标记
        marker = markers[i % len(markers)] if len(unique_labels) > 6 else 'o'
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    s=point_size, alpha=alpha,
                    color=palette[i],
                    marker=marker,
                    label=str(label))

    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_colored_with_shapes(embedding, labels, point_size, alpha):
    """中等类别数的着色方案（颜色+形状）"""
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", min(len(unique_labels), 12))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X', 'h', '+']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color_idx = i % len(palette)
        marker_idx = (i // len(palette)) % len(markers)
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    s=point_size, alpha=alpha,
                    color=palette[color_idx],
                    marker=markers[marker_idx],
                    label=str(label))

    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


def plot_continuous(embedding, labels, point_size, alpha):
    """大量类别时的连续着色方案"""
    print(f"⚠️ 检测到大量类别（{len(np.unique(labels))}），使用连续色谱")
    plt.scatter(embedding[:, 0], embedding[:, 1],
                s=point_size, alpha=alpha,
                c=labels, cmap='viridis')
    plt.colorbar(label='Label Values')

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import random


def extract_features_with_sampling(model, loader, max_samples=2000, verbose=False):
    """
    从测试集中提取特征并进行随机采样

    参数:
    model: 训练好的模型
    loader: 数据加载器
    max_samples: 最大采样样本数
    verbose: 是否显示进度条

    返回:
    包含特征张量和标签的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 初始化特征存储
    all_features = {
        'layer4': [],
        'mul': [],
        'std': [],
        'fe': [],
        'labels': []
    }

    # 存储所有批次的特征
    full_features = {
        'layer4': [],
        'mul': [],
        'std': [],
        'fe': [],
        'labels': []
    }

    with torch.no_grad():
        for batch_sample in tqdm(iter(loader), desc="Extracting features",
                                 total=len(loader), disable=not verbose):
            # 提取输入数据和标签
            img_rgb = batch_sample['image_x'].to(device)
            img_ir = batch_sample['image_ir'].to(device)
            img_depth = batch_sample['image_depth'].to(device)
            labels = batch_sample['binary_label'].to(device)

            # 前向传播获取特征
            output, p, mul, std, x_m, layer4,s = model(img_rgb, img_ir, img_depth)

            # 计算fe特征
            fe = mul + std

            # 收集特征数据 - 保持原始维度
            full_features['layer4'].append(layer4.cpu())
            full_features['mul'].append(mul.cpu())
            full_features['std'].append(std.cpu())
            full_features['fe'].append(fe.cpu())
            full_features['labels'].append(labels.cpu())

    # 合并所有批次的数据
    for key in full_features:
        if full_features[key]:
            full_features[key] = torch.cat(full_features[key], dim=0)

    # 采样逻辑
    total_samples = full_features['labels'].shape[0]
    if max_samples < total_samples:
        # 确保每个类别按比例采样
        class_0_idx = torch.where(full_features['labels'] == 0)[0]
        class_1_idx = torch.where(full_features['labels'] == 1)[0]

        # 按类别比例计算采样数量
        class_0_samples = int(max_samples * len(class_0_idx) / total_samples)
        class_1_samples = int(max_samples * len(class_1_idx) / total_samples)
        total_samples = class_0_samples + class_1_samples

        # 随机采样
        sampled_class_0 = random.sample(class_0_idx.tolist(), min(class_0_samples, len(class_0_idx)))
        sampled_class_1 = random.sample(class_1_idx.tolist(), min(class_1_samples, len(class_1_idx)))
        sampled_indices = torch.tensor(sampled_class_0 + sampled_class_1)

        # 使用采样索引提取特征
        for key in all_features:
            all_features[key] = full_features[key][sampled_indices]
    else:
        # 如果没有超过最大样本数，则使用全部数据
        all_features = full_features

    print(f"Selected {len(all_features['labels'])} samples for t-SNE visualization.")
    return all_features


def visualize_tsne_for_feature(feature, labels, save_path, feature_name, perplexity=20):
    """
    绘制单个特征的t-SNE图并保存

    参数:
    feature: 特征张量 (B, C, H, W)
    labels: 标签张量 (B,)
    save_path: 图像保存完整路径（包含文件名）
    feature_name: 特征名称（用于标题）
    perplexity: t-SNE复杂度参数
    """
    # 准备特征数据
    feature = feature.view(feature.size(0), -1).numpy()  # 展平为(B, C*H*W)
    labels = labels.numpy()

    # PCA预处理（当维度>100时）
    if feature.shape[1] > 100:
        pca = PCA(n_components=min(50, feature.shape[1]))
        feature = pca.fit_transform(feature)

    # 特征标准化
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature)

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000, learning_rate=200)
    embedding = tsne.fit_transform(feature_scaled)

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 分离不同类别
    class_0_mask = (labels == 0)
    class_1_mask = (labels == 1)

    # 绘制散点图
    plt.scatter(embedding[class_0_mask, 0], embedding[class_0_mask, 1],
                color='#1f77b4', alpha=0.7, s=40, label='Class 0 (Real)',
                edgecolors='w', linewidths=0.5)

    plt.scatter(embedding[class_1_mask, 0], embedding[class_1_mask, 1],
                color='#d62728', alpha=0.7, s=40, label='Class 1 (Fake)',
                edgecolors='w', linewidths=0.5)

    # 添加标题和图例
    title_name = {
        'layer4': 'CNN Layer 4 Features',
        'mul': 'Multiplication Features',
        'std': 'Standard Deviation Features',
        'fe': 'Combined (mul+std) Features'
    }.get(feature_name, feature_name)

    plt.title(f't-SNE Visualization of {title_name}', fontsize=16, pad=15)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)

    # 移除坐标轴和网格线
    plt.axis('off')

    # 添加文本标签
    plt.text(0.99, 0.01, f'n={len(labels)}, perplexity={perplexity}',
             transform=plt.gca().transAxes, fontsize=10,
             horizontalalignment='right', verticalalignment='bottom')

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")


def visualize_all_features_tsne(model, loader, save_path_template, max_samples=2000, epoch=0, batch_num=0,
                                verbose=False):
    """
    提取特征并绘制所有特征的t-SNE图

    参数:
    model: 训练好的模型
    loader: 数据加载器
    save_path_template: 图像保存路径模板，包含{feature}占位符
    max_samples: 最大采样样本数
    epoch: 当前epoch（用于文件名）
    batch_num: 当前batch编号（用于文件名）
    verbose: 是否显示进度条
    """
    # 提取特征并进行采样
    features_dict = extract_features_with_sampling(model, loader, max_samples, verbose)

    # 为每个特征绘制t-SNE图
    feature_names = ['layer4', 'mul', 'std', 'fe']

    for feat_name in feature_names:
        if feat_name in features_dict and features_dict[feat_name].numel() > 0:
            # 构建特定特征的保存路径
            save_path = save_path_template.format(
                epoch=epoch,
                batch=batch_num,
                feature=feat_name
            )

            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 可视化单个特征
            visualize_tsne_for_feature(
                feature=features_dict[feat_name],
                labels=features_dict['labels'],
                save_path=save_path,
                feature_name=feat_name
            )
        else:
            print(f"Warning: Feature '{feat_name}' not found or empty. Skipping.")

    return features_dict

def modality_drop(x_rgb, x_ir, x_depth, p, args):
    # print(p)
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    # p=[1,1,1]
    index_list = [x for x in range(7)]

    if p == [0, 0, 0]:
        # print("drop")
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array(( 1 / 7, 1 / 7,1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_ir = x_ir * p[:, 1]
    x_depth = x_depth * p[:, 2]
    p = p.squeeze()

    return x_rgb, x_ir, x_depth, p


def modality_drop_p(x_rgb, x_ir, x_depth, p, args):
    """
    对RGB、红外(IR)和深度(Depth)模态应用模态丢弃(Modality Dropout)[2](@ref)。
    输入张量形状应为 (B, C, H, W)，例如 (B, 3, 112, 112)。

    参数:
        x_rgb: RGB模态输入张量
        x_ir: 红外模态输入张量
        x_depth: 深度模态输入张量
        p: 预定义的模态保留概率列表，如 [1,1,1] 表示全部保留。
           如果为 [0,0,0]，则从7种固定组合中随机选择。
        args: 包含其他参数的命名元组或字典（用于日志记录等）

    返回:
        x_rgb: 应用模态丢弃后的RGB张量
        x_ir: 应用模态丢弃后的红外张量
        x_depth: 应用模态丢弃后的深度张量
        p: 实际使用的掩码张量（形状为(B, 3)），便于后续分析
    """
    # 7种可能的模态组合[2](@ref): [RGB, IR, Depth]
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 0], [1, 0, 1], [0, 1, 1],
                            [1, 1, 1]]
    index_list = [x for x in range(7)]  # 组合索引列表

    if p == [0, 0, 0]:
        # 随机选择模态组合[2](@ref)
        p = []
        # 为批次中的每个样本随机选择一种模态组合
        prob = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])  # 均匀概率
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # 如果需要记录索引（例如用于日志记录或分析），可以在此处使用args中的写入器
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p)
        p = torch.from_numpy(p)

    else:
        # 使用给定的概率p，并扩展到整个批次
        p = p
        p = [p] * x_rgb.shape[0]  # 为批次中的每个样本使用相同的概率
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)

    # 将p转换为FloatTensor并移动到GPU（如果输入张量在GPU上）
    p = p.float().to(x_rgb.device)

    # 调整掩码维度以匹配输入张量 (B, 3, 112, 112)
    # 从 (B, 3) 扩展为 (B, 3, 1, 1)，以便通过广播与 (B, 3, 112, 112) 相乘
    p_expanded = p.unsqueeze(-1).unsqueeze(-1)  # 现在形状是 (B, 3, 1, 1)

    # 对每个模态应用掩码
    x_rgb = x_rgb * p_expanded[:, 0]  # 使用RGB对应的掩码（第0列）
    x_ir = x_ir * p_expanded[:, 1]  # 使用IR对应的掩码（第1列）
    x_depth = x_depth * p_expanded[:, 2]  # 使用Depth对应的掩码（第2列）

    # 返回处理后的张量和掩码（掩码p的形状为(B, 3)）
    return x_rgb, x_ir, x_depth, p

def modality_drop_v(x_rgb, x_ir, x_depth, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    index_list = [x for x in range(7)]

    # 创建输入数据的副本，以便分别处理原始缺失和对立缺失
    x_rgb_orig = x_rgb.clone()
    x_ir_orig = x_ir.clone()
    x_depth_orig = x_depth.clone()

    if p == [0, 0, 0]:
        # 随机生成缺失模式
        prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        p_list = []
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p_list.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p_list)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        # 重用给定的缺失模式
        p = [p] * x_rgb.shape[0]  # 复制到batch中每个样本
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    # 应用原始缺失模式到输入数据
    x_rgb_original_drop = x_rgb_orig * p[:, 0]
    x_ir_original_drop = x_ir_orig * p[:, 1]
    x_depth_original_drop = x_depth_orig * p[:, 2]

    # 准备返回的p (移除多余的维度)
    p_return = p.squeeze()

    # 计算对立缺失模式q (0->1, 1->0)
    q = 1 - p_return
    # 特殊情况处理：当原始模式是[1,1,1]时，对立模式也设为[1,1,1]
    all_ones = torch.all(p_return == 1, dim=1)
    q[all_ones] = p_return[all_ones]

    # 扩展q的维度以匹配输入数据的维度
    q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    # 应用对立缺失模式到输入数据
    x_rgb_opposite_drop = x_rgb_orig * q_expanded[:, 0]
    x_ir_opposite_drop = x_ir_orig * q_expanded[:, 1]
    x_depth_opposite_drop = x_depth_orig * q_expanded[:, 2]

    # 返回两组数据：原始缺失处理后的数据和对立缺失处理后的数据
    return  x_rgb_original_drop, x_ir_original_drop, x_depth_original_drop,x_rgb_opposite_drop, x_ir_opposite_drop, x_depth_opposite_drop,  p_return, q


def modality_drop1(x_rgb, x_ir, x_depth, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]

    if p == [0, 0, 0]:
        # print("drop")
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array(( 11/42, 2/21,11/42, 2/21, 2/21, 2/21, 2/21))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_ir = x_ir * p[:, 1]
    x_depth = x_depth * p[:, 2]

    return x_rgb, x_ir, x_depth, p
def unbalance_modality_drop(x_rgb, x_ir, x_depth, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]
    prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
    # print(args.epoch)
    mode_num = 7
    hard_mode_index = [0, 2, 4]
    mode_average = x_rgb.shape[0] // mode_num
    batch_left = x_rgb.shape[0] % mode_num
    mode_left = 2
    if p == [0, 0, 0]:
        p = []
        # prob = np.array([3 / 12, 1 / 12, 3 / 12, 1 / 12, 2 / 12, 1 / 12, 1 / 12])
        # for i in range(x_rgb.shape[0]):
        #     index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
        #     p.append(modality_combination[index])
        #     # if 'model_arch_index' in args.writer_dicts.keys():
        #     #     args.writer_dicts['model_arch_index'].write(str(index) + " ")
        #
        # p = np.array(p)
        # p = torch.from_numpy(p)
        # p = torch.unsqueeze(p, 2)
        # p = torch.unsqueeze(p, 3)
        # p = torch.unsqueeze(p, 4)

        if args.epoch < 15:
            for i in range(mode_num):
                p = p + modality_combination[i] * mode_average
            for i in range(batch_left):
                p = p + modality_combination[i]
        else:
            increase_num =  args.epoch - 15
            if increase_num > 7:
                increase_num = 7

            # print(increase_num)
            for i in hard_mode_index:
                p = p + modality_combination[i] * (mode_average + increase_num)

            decrease_num = args.epoch - 15
            if decrease_num > 7:
                decrease_num = 7

            # print(decrease_num)
            for i in [3,5,6]:
                p = p + modality_combination[i] * (mode_average - decrease_num)
            p=p + modality_combination[1] * mode_average
            for i in range(batch_left):
                p = p + modality_combination[i]

        # p = p + modality_combination[2] * 17
        # for i in [0, 4]:
        #     p = p + modality_combination[i] * 11
        # for i in [1, 3, 5]:
        #     p = p + modality_combination[i] * 7
        # p = p + modality_combination[6] * 4
        p = np.array(p)
        p = p.reshape((64, 3))
        np.random.shuffle(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)



    else:
        p = p
        p = [p * x_rgb.shape[0]]
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_ir = x_ir * p[:, 1]
    x_depth = x_depth * p[:, 2]

    return x_rgb, x_ir, x_depth, p
