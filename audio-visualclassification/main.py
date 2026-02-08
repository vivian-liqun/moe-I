import argparse

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 Python 路径
sys.path.append(current_dir)
from tqdm import tqdm
import torch
print(1)
print(torch.cuda.is_available())
import torch.nn as nn
print(1)
print(torch.cuda.is_available())
import torch.optim as optim
print(1)
print(torch.cuda.is_available())
from torch.utils.data import DataLoader
print(1)
print(torch.cuda.is_available())
from torch.utils.tensorboard import SummaryWriter
print(1)
print(torch.cuda.is_available())
import torch.nn.functional as F
import pdb
print(1)
print(torch.cuda.is_available())
from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from dataset.KSDataset import KSDataset
from models.basic_model import AVClassifier,AVClassifier_teacher,ACVWithPrompt,AVClassifier_valid
from utils.utils import setup_seed, weight_init
import csv
import numpy as np
print(1)
print(torch.cuda.is_available())
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KineticSound', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Normal', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=1, type=int)
    parser.add_argument('--audio_path', default='.\data\CREMA-D\AudioWAV', type=str)
    parser.add_argument('--visual_path', default='.\data\CREMA-D', type=str)
    parser.add_argument('--pre_teacher_root', type=str, default='.\pre_teacher\\Unif_T_ks.pth')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')

    parser.add_argument('--ckpt_path',  type=str,default='./results/ks/pme_share', help='path to save trained models')
    parser.add_argument('--train', action='store_true', default=True, help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pme', type=int, default=1)

    return parser.parse_args()

def get_feature_diversity(a_feature):
    a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
    a_feature = a_feature.permute(0, 2, 1)  # B HW C
    a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
    a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
    a_std = torch.std(a_feature, dim=2)
    a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
    a_similarity = a_similarity / a_std_matrix
    # print(a_similarity)
    a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
    # print(a_norm.shape)
    a_norm = torch.mean(a_norm)
    return a_norm


def regurize(mul, std):
    variance_dul = std ** 2
    variance_dul = variance_dul.view(variance_dul.shape[0], -1)
    mul = mul.view(mul.shape[0], -1)
    loss_kl = torch.sum(((variance_dul + mul ** 2 - torch.log(variance_dul) - 1) * 0.5), dim=1)
    loss_kl = torch.mean(loss_kl)

    return loss_kl


def get_feature_diff(x1, x2):
    # print(x1.shape,x2.shape)
    x1 = F.adaptive_avg_pool2d(x1, (7, 7))
    x2 = F.adaptive_avg_pool2d(x2, (7, 7))
    # x1 = torch.mean(x1, dim=(2, 3))
    # x2 = torch.mean(x2, dim=(2, 3))

    x1 = x1.permute(0, 2, 3, 1).contiguous()
    x2 = x2.permute(0, 2, 3, 1).contiguous()

    rgb = x1.view(-1, x1.shape[3])
    depth = x2.view(-1, x2.shape[3])

    diff = F.mse_loss(rgb, depth)
    # diff = torch.cosine_similarity(rgb, depth)
    # diff = torch.mean(diff)
    # print(simi.shape)
    return diff

def normalize_tensor(tensor):
    """
    对输入张量进行归一化
    """
    # 检查张量的维度
    num_dims = len(tensor.shape)
    if num_dims < 4:
        # 如果维度小于 4，在所有非批量维度上计算均值和标准差
        dims = tuple(range(1, num_dims))
    else:
        dims = (1, 2, 3)
    # 计算均值和标准差
    mean = tensor.mean(dim=dims, keepdim=True)
    std = tensor.std(dim=dims, keepdim=True)
    # 归一化
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # 避免除零
    return normalized_tensor
def samplewise_cosine_similarity(mul, std):
    """
    逐样本计算余弦相似度
    """
    # 归一化输入张量
    mul = normalize_tensor(mul)
    std = normalize_tensor(std)
    # 检查张量的维度
    num_dims = len(mul.shape)
    if num_dims < 4:
        dims = tuple(range(1, num_dims))
    else:
        dims = (1, 2, 3)
    # 计算点积
    dot_product = (mul * std).sum(dim=dims)
    # 计算 L2 范数
    mul_norm = torch.norm(mul, p=2, dim=dims)
    std_norm = torch.norm(std, p=2, dim=dims)
    # 计算余弦相似度
    cosine_sim = dot_product / (mul_norm * std_norm + 1e-10)
    return cosine_sim
def compute_samplewise_cosine_loss(mul, std):
    """
    计算逐样本余弦相似度损失
    """
    cosine_sim = samplewise_cosine_similarity(mul, std)
    # 将余弦相似度映射到 [0, 1] 范围
    positive_similarity = (cosine_sim ) / 2
    loss = positive_similarity.sum()  # 将所有样本的相似度相加
    return loss

def compute_samplewise_cosine_loss_p(mul, std):
    """
    计算逐样本余弦相似度损失
    """
    cosine_sim = samplewise_cosine_similarity(mul, std)
    # 将余弦相似度映射到 [0, 1] 范围
    positive_similarity = (1-cosine_sim ) / 2
    loss = positive_similarity.sum()  # 将所有样本的相似度相加
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def train_epoch(args, differences,epoch, model,tp_model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    model.train()
    print("Start training ... ")

    train_loss = 0
    train_loss_cl = 0
    train_loss_m = 0
    train_loss_d = 0
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0


    similar_average = 0
    std_a_sum = 0
    std_vum = 0
    total_std = 0
    total_mul = 0

    if epoch > 20:

        differences=differences
        w = [1, 1, 1]
    else:
        differences = [0,0,0]
        w=[0,0,0]

    print(f"梯度使用的权重为: {differences}")
    print(f"提示使用的权重为: {w}")

    for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Train Data Loading Progress")):
    # for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        teacher_model=tp_model

        a, v, out, p,layer1,layer2,layer3  = model(spec.unsqueeze(1).float(), image.float())
        a_t, v_t, out_t, p_t,layer1_t,layer2_t,layer3_t = teacher_model(spec.unsqueeze(1).float(), image.float(),p,w)



        loss_cls_m = criterion(out, label)

        e_loss1 = compute_samplewise_cosine_loss_p(layer1, layer1_t)
        e_loss2 = compute_samplewise_cosine_loss_p(layer2, layer2_t)
        e_loss3 = compute_samplewise_cosine_loss_p(layer3, layer3_t)
        e_loss4 = compute_samplewise_cosine_loss_p(out, out_t)
        e_loss = e_loss2 + e_loss1 + e_loss4 + e_loss3

        if epoch > 5:

            loss_cls =loss_cls_m*14
        else:
            loss_cls = loss_cls_m*14

        loss = loss_cls+e_loss

        loss.backward()
        if isinstance(differences, torch.Tensor):
            differences = differences.clone().detach()
        else:
            # 如果differences原本是Python列表，转换为张量
            differences = torch.tensor(differences)
        prompt_weights_tensor = differences.cuda()

        if teacher_model.prompt_layer3.grad is not None:
            with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                # 遍历每个提示类型 (0到6)
                for i in range(teacher_model.prompt_layer3.grad.size(0)):  # 第一个维度是提示数量
                    if teacher_model.prompt_layer3.grad is not None:
                        # 将第 i 个提示的梯度乘以其对应的权重
                        teacher_model.prompt_layer3.grad[i] = teacher_model.prompt_layer3.grad[i] * prompt_weights_tensor[i]

        if teacher_model.prompt_layer2.grad is not None:
            with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                # 遍历每个提示类型 (0到6)
                for i in range(teacher_model.prompt_layer2.grad.size(0)):  # 第一个维度是提示数量
                    if teacher_model.prompt_layer2.grad is not None:
                        # 将第 i 个提示的梯度乘以其对应的权重
                        teacher_model.prompt_layer2.grad[i] = teacher_model.prompt_layer2.grad[i] * prompt_weights_tensor[i]

        if teacher_model.prompt_layer1.grad is not None:
            with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                # 遍历每个提示类型 (0到6)
                for i in range(teacher_model.prompt_layer1.grad.size(0)):  # 第一个维度是提示数量
                    if teacher_model.prompt_layer1.grad is not None:
                        # 将第 i 个提示的梯度乘以其对应的权重
                        teacher_model.prompt_layer2.grad[i] = teacher_model.prompt_layer1.grad[i] * prompt_weights_tensor[i]

        optimizer.step()

        train_loss_d += loss_cls.item()
        train_loss_cl += loss_cls.item()

        train_loss += loss.item()
        # loss_kl_sum += loss_kl.item()
        _loss += loss.item()
        _loss_v += loss_cls.item()


    scheduler.step()
    print(
        "Epoch {}, loss={:.5f}, loss_m={:.5f},loss_d={:.5f},loss_cl={:.5f}".format(
            epoch, train_loss / len(dataloader),
                    train_loss_m / len(dataloader),
                   train_loss_d / len(dataloader), train_loss_cl / len(dataloader)))


    print("std,mul", total_std / len(dataloader), total_mul / len(dataloader))



    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader)


def valid_g(args, model, device, dataloader):
    gpu_ids = list(range(torch.cuda.device_count()))
    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    teacher_model = AVClassifier_valid(args)
    model_path = args.pre_teacher_root
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpu_ids)
    teacher_model.load_state_dict(torch.load(model_path)['model'], strict=True)
    teacher_model.eval()
    teacher_model.cuda()

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        total_loss_rgb = 0
        total_loss_ir = 0
        total_loss_depth = 0
        total_loss_rgb_t = 0
        total_loss_ir_t = 0
        total_loss_depth_t = 0

        # for step, (spec, image, label) in enumerate(dataloader):
        for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Valid Data Loading Progress")):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)


            a, v, out,p,x,y,j = model(spec.unsqueeze(1).float(), image.float())
            a_t, v_t, out_t, p_t, x_t, y_t, j_t = teacher_model(spec.unsqueeze(1).float(), image.float(),p)
            cond_rgb_all = (p[:, 0] * (1 - p[:, 1]) ) == 1
            cond_ir_all = ((1 - p[:, 0]) * p[:, 1] ) == 1
            cond_depth_all = ((p[:, 0]) * (p[:, 1]) ) == 1
            auxi_loss_m = auxi_cross_entropy(out, label)
            auxi_loss_t = auxi_cross_entropy(out_t, label)
            epsilon = 0.1

            # 计算各种模态的损失
            x_rgb = torch.sum(auxi_loss_m * (p[:, 0] * (1 - p[:, 1]))) / (
                    cond_rgb_all.sum() + epsilon)
            x_ir = torch.sum(auxi_loss_m * ((1 - p[:, 0]) * p[:, 1] )) / (
                    cond_ir_all.sum() + epsilon)
            x_depth = torch.sum(auxi_loss_m * ((p[:, 0]) * (p[:, 1]) )) / (
                    cond_depth_all.sum() + epsilon)
            x_rgb_t = torch.sum(auxi_loss_t * (p[:, 0] * (1 - p[:, 1]))) / (
                    cond_rgb_all.sum() + epsilon)
            x_ir_t = torch.sum(auxi_loss_t * ((1 - p[:, 0]) * p[:, 1])) / (
                    cond_ir_all.sum() + epsilon)
            x_depth_t = torch.sum(auxi_loss_t * ((p[:, 0]) * (p[:, 1]))) / (
                    cond_depth_all.sum() + epsilon)
            total_loss_rgb += x_rgb.item()
            total_loss_ir += x_ir.item()
            total_loss_depth += x_depth.item()

            total_loss_rgb_t += x_rgb_t.item()
            total_loss_ir_t += x_ir_t.item()
            total_loss_depth_t += x_depth_t.item()
        print(
            " total_loss_a={:.5f}, total_loss_v={:.5f}, total_loss_av={:.5f}".format(
                total_loss_rgb, total_loss_ir, total_loss_depth,

            )
        )
        print(
            " total_loss_a_t={:.5f}, total_loss_v_t={:.5f}, total_loss_av_t={:.5f}".format(
                total_loss_rgb_t, total_loss_ir_t, total_loss_depth_t,

            )
        )
        d_rgb = total_loss_rgb - total_loss_rgb_t
        d_ir = total_loss_ir - total_loss_ir_t
        d_depth = total_loss_depth - total_loss_depth_t
        differences = [d_rgb, d_ir, d_depth]
        differences = np.array(differences)
        differences = differences
        exp_x = np.exp(differences)

        # 计算指数
        probabilities = exp_x / np.sum(exp_x)  # 归一化[1,6](@ref)
        prob = np.array(probabilities)
        print(f"适配度: {prob}")


    return prob,differences

def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        # for step, (spec, image, label) in enumerate(dataloader):
        for step, (spec, image, label) in enumerate(tqdm(dataloader, desc="Valid Data Loading Progress")):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out,p,x,y,j = model(spec.unsqueeze(1).float(), image.float())


            prediction = softmax(out)
            # pred_v = softmax(out_v)
            # pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                # v = np.argmax(pred_v[i].cpu().data.numpy())
                # a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0


    return sum(acc) / sum(num), sum(acc) / sum(num), sum(acc) / sum(num)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    args = get_arguments()
    print(args)

    seed_torch(3)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    args.p = [0, 0]
    args.weak_modality = 'none'

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model_path = args.pre_teacher_root
    model_path = "D:\YJH\CED\\audio-visualclassification\pre_teacher\\Unif_T_ks.pth"
    model_t = AVClassifier_teacher(args)
    model_t.to(device)

    model_t = torch.nn.DataParallel(model_t, device_ids=gpu_ids)
    model_t.load_state_dict(torch.load(model_path)['model'], strict=True)

    model_t.cuda()

    tp_model = ACVWithPrompt(model_t, args)

    all_parameters = list(model.parameters()) + list(tp_model.parameters())

    model.cuda()
    tp_model.cuda()

    optimizer = optim.SGD(all_parameters, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = KSDataset(args, mode='train')
        test_dataset = KSDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    #
    # acc, acc_a, acc_v = valid(args, model_t, device, test_dataloader)
    # print('Accuracyt: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))

    if args.train:

        best_acc = 0.0

        std_a_sum = 0
        std_v_sum = 0
        differences=[0,0,0]

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, std_a, std_v = train_epoch(args,
                                                                                    scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)


                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy = train_epoch(args,differences,
                                                                                                             epoch,
                                                                                                             model,tp_model,
                                                                                                             device,
                                                                                                             train_dataloader,
                                                                                                             optimizer,
                                                                                                             scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                # differences,dp = valid_g(args, model, device, test_dataloader)
                # d = [-x for x in dp]
                # print(f"模态匹配度: {d}")
                # pretrain_dir = "D:\YJH\ECCV2024-DMRNet-main\\audio-visualclassification/results/ks/pme_share_miss_cmad/md"
                # directory = os.path.dirname(pretrain_dir)
                # # 创建目录（如果不存在），exist_ok=True 确保目录已存在时不会报错
                # os.makedirs(directory, exist_ok=True)
                # with open(pretrain_dir, 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([epoch] + d)







            if acc > best_acc and epoch:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                model_name = 'best_model_of_dataset_{}_{}_beta_{}_pe_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_' \
                             'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                          args.modulation,
                                                          args.beta,
                                                          args.pe,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(batch_loss_a, batch_loss_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
                print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(batch_loss_a, batch_loss_v))

    else:
        # first load trained model
        os.makedirs(args.ckpt_path, exist_ok=True)
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
        # print(state_dict)
        model.load_state_dict(state_dict)
        # model.train()
        # model.eval()
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()